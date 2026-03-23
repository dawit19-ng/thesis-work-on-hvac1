"""
hvac_energy_forecasting.py

This script implements deep learning models (LSTM, BiLSTM, GRU, SSM-inspired)
to forecast HVAC chiller energy consumption using hourly time-series data.

 It includes:
- Data loading and preprocessing
- Feature engineering (temporal, lag, rolling statistics)
-Sequencecreation for LSTM/GRU models
- Model definitions
- Training with early stoping and learning rate scheduler
- Evaluation metrics (RMSE, MAE, R², MAPE, Accuracy)
- Baseline models(Random Forest, XGBoost)
- Ensemble predictions and visualization

Authors: Abrha Dawit Nigusse
 
"""

# ================================================================
#  Chiller Forcasting- LSTM / BiLSTM/ GRU / CPU-ONLY SSM model
#  (Zero external deps – pure PyTorch )

# ================================================================
