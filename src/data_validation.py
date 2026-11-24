import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series


class OutpatientSchema(pa.DataFrameModel):
    lead_time_days: Series[int] = pa.Field(ge=0, le=365)
    age_band: Series[int] = pa.Field(isin=[1, 2, 3, 4])  # 1:18-30, 2:31-50, 3:51-70, 4:71+
    imd_decile: Series[int] = pa.Field(ge=1, le=10)      # 1 = Most deprived, 10 = Least deprived
    prev_dna_count: Series[int] = pa.Field(ge=0, le=50)
    prev_attended_count: Series[int] = pa.Field(ge=0, le=100)
    is_morning_slot: Series[int] = pa.Field(isin=[0, 1])
    distance_km: Series[float] = pa.Field(ge=0.0, le=200.0)
    specialty_code: Series[int] = pa.Field(isin=[100, 110, 300, 400, 500])  # NHS Treatment Function Codes
    did_not_attend: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        coerce = True


def generate_synthetic_nhs_data(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generates privacy-safe synthetic NHS outpatient records."""
    rng = np.random.default_rng(seed)
    lead_time = rng.integers(1, 120, size=n_rows)
    age_band = rng.choice([1, 2, 3, 4], p=[0.20, 0.30, 0.30, 0.20], size=n_rows)
    imd_decile = rng.integers(1, 11, size=n_rows)
    prev_dna = rng.poisson(0.4, size=n_rows)
    prev_att = rng.poisson(3.0, size=n_rows)
    morning = rng.integers(0, 2, size=n_rows)
    distance = np.round(rng.gamma(2.0, 4.5, size=n_rows), 1)
    specialty = rng.choice([100, 110, 300, 400, 500], size=n_rows)

    # Realistic underlying risk signal
    logit = (
        -2.6
        + 0.015 * lead_time
        + 0.55 * prev_dna
        - 0.12 * prev_att
        + 0.25 * (age_band == 1).astype(int)
        + 0.03 * distance
        - 0.04 * imd_decile
    )
    prob = 1 / (1 + np.exp(-logit))
    dna = rng.binomial(1, prob)

    df = pd.DataFrame({
        "lead_time_days": lead_time,
        "age_band": age_band,
        "imd_decile": imd_decile,
        "prev_dna_count": prev_dna,
        "prev_attended_count": prev_att,
        "is_morning_slot": morning,
        "distance_km": distance,
        "specialty_code": specialty,
        "did_not_attend": dna,
    })
    return OutpatientSchema.validate(df)
