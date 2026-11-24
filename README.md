# NHS-DNA-Trustworthy_MLOps

A Trustworthy-MLOps reference implementation for predicting outpatient appointment non-attendance (DNA) in NHS settings.

The project combines synthetic data generation, data validation, calibrated machine-learning predictions, Fairlearn fairness auditing, SHAP explainability, MLflow model registration, Azure infrastructure, Docker, and a FastAPI scoring service.

> **Safety notice:** This system is for decision support and testing only. It must not be used to cancel appointments or make automated clinical decisions.

## Features

- Synthetic, privacy-safe NHS outpatient data generation
- Pandera data validation
- Calibrated LightGBM DNA-risk prediction
- Fairlearn deprivation-group fairness audit
- Automated performance and fairness governance gate
- SHAP local explanations
- MLflow model logging and registration
- FastAPI scoring API
- Docker container support
- Azure Bicep infrastructure
- UK South deployment configuration

## Repository Structure

```text
.github/workflows/        GitHub Actions workflow
docs/                     Clinical safety and algorithmic transparency documents
infra/
  main.bicep              Azure infrastructure definition
pipelines/
  azure_ml_pipeline.yml   Azure Machine Learning pipeline
src/
  data_validation.py      Synthetic data generation and validation
  train_and_evaluate.py   Training, auditing, and model registration
  score_api.py            FastAPI prediction service
Dockerfile                Container configuration
requirements.txt          Python dependencies

Architecture

Synthetic/validated data
          |
          v
LightGBM training
          |
          v
Calibration + performance metrics
          |
          v
Fairness audit across deprivation groups
          |
          +--> Governance failure: model blocked
          |
          +--> Governance pass: model registered
                            |
                            v
                    FastAPI scoring service
                            |
                            v
                    Risk tier + SHAP explanations



Prerequisites

Python 3.11 or later
Azure CLI for Azure deployments
Docker for containerized execution
An Azure subscription with appropriate permissions
MLflow for local model tracking