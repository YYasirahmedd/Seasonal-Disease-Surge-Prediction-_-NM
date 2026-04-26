# Seasonal Disease Surge Prediction
## Early Warning System for Dengue Outbreaks

**Hackathon:** TNSDC Naan Mudhalvan 2026 Advanced AI/ML  
**Problem Statement:** PS 08  
**Institution:** Sriram Engineering College, Chennai

---

## Project Overview
This project builds an expert hybrid ML+DL model that predicts whether a dengue fever surge will occur in the coming week for any Sri Lankan district. The system uses historical dengue case data combined with lagged environmental features (rainfall, temperature, humidity) to give health authorities 1-2 weeks early warning before an outbreak peaks.

## Model Architecture
- **Model 1:** GRU (Deep Learning) — learns time-series patterns
- **Model 2:** Random Forest — learns feature interactions
- **Model 3:** XGBoost — gradient boosting on residuals
- **Meta-Learner:** Logistic Regression stacking layer

## Key Results
| Metric | Baseline | Hybrid Model | Improvement |
|--------|----------|--------------|-------------|
| Recall | 0.28 | 0.50 | +78.6% |
| F1 Score | 0.22 | 0.35 | +59.1% |
| ROC-AUC | 0.54 | 0.69 | +27.8% |

## How to Run
1. Open notebooks/dengue_expert_hybrid.ipynb in Google Colab
2. Enable T4 GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom (Runtime → Run All)
4. Upload dengue_final_dataset.csv when prompted
5. Expected runtime: ~5 minutes on GPU

## Dataset
Place dengue_final_dataset.csv in the /data folder.  
25 Sri Lankan districts, 2019-2021, 789 usable rows after cleaning.

## Requirements
See requirements.txt for full dependency list.
Key: tensorflow==2.15.0, xgboost==2.0.3, scikit-learn==1.3.0, imbalanced-learn==0.11.0
