import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from fairlearn.metrics import MetricFrame, equalized_odds_difference, false_negative_rate
from data_validation import generate_synthetic_nhs_data

FAIRNESS_THRESHOLD = 0.06  # Max allowed Equalized Odds difference across deprivation groups


def train():
    mlflow.autolog(disable=True)
    with mlflow.start_run(run_name="nhs-dna-trustworthy-build") as run:
        df = generate_synthetic_nhs_data(n_rows=15000)

        features = [
            "lead_time_days", "age_band", "imd_decile", "prev_dna_count",
            "prev_attended_count", "is_morning_slot", "distance_km", "specialty_code"
        ]
        X = df[features]
        y = df["did_not_attend"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 1. Base LightGBM + Isotonic Calibration for accurate probability risk scores
        base_clf = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
        )
        calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        calibrated_clf.fit(X_train, y_train)

        # Also fit standalone base estimator on full train set for exact TreeSHAP explanations
        base_clf.fit(X_train, y_train)

        probs = calibrated_clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.25).astype(int)  # Operational outreach threshold

        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        # 2. Trustworthy AI Fairness Audit across Deprivation Groups (IMD 1-3 vs 4-10)
        deprived_group = np.where(X_test["imd_decile"] <= 3, "High_Deprivation_IMD_1_3", "Other_IMD_4_10")
        eo_diff = equalized_odds_difference(y_test, preds, sensitive_features=deprived_group)

        mf = MetricFrame(
            metrics={"fnr": false_negative_rate},
            y_true=y_test,
            y_pred=preds,
            sensitive_features=deprived_group,
        )

        mlflow.log_metrics({
            "roc_auc": round(auc, 4),
            "brier_calibration_score": round(brier, 4),
            "equalized_odds_diff_imd": round(eo_diff, 4),
        })

        # 3. Automated Governance Gate
        governance_report = {
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "equalized_odds_diff_imd": round(eo_diff, 4),
            "fnr_by_deprivation": mf.by_group["fnr"].to_dict(),
            "passed_trustworthy_gate": bool(eo_diff <= FAIRNESS_THRESHOLD and auc >= 0.72),
        }

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/governance_audit.json", "w") as f:
            json.dump(governance_report, f, indent=2)
        mlflow.log_artifact("outputs/governance_audit.json")

        if not governance_report["passed_trustworthy_gate"]:
            raise RuntimeError(f"BLOCKED BY TRUSTWORTHY AI GATE: {governance_report}")

        # 4. Save artifacts & register model in Azure ML Registry
        bundle = {"calibrated_model": calibrated_clf, "explainer_model": base_clf, "features": features}
        joblib.dump(bundle, "outputs/model_bundle.joblib")
        mlflow.log_artifact("outputs/model_bundle.joblib")

        mlflow.sklearn.log_model(
            sk_model=calibrated_clf,
            artifact_path="nhs_dna_model",
            registered_model_name="nhs-outpatient-dna-risk",
        )
        print("Model passed governance checks and registered:", governance_report)


if __name__ == "__main__":
    train()
