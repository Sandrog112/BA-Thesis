import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import logging

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

train_log_file = os.path.join(log_dir, 'train.log')

logging.basicConfig(
    filename=train_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing import prepare_data

def train_xgboost(data, commodity_name, params=None):
    """Train XGBoost model for a specific commodity"""
    logging.info(f"Starting training for {commodity_name}")
    train_X = data['train_X']
    test_X = data['test_X']
    train_y = data['train_y']
    test_y = data['test_y']

    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 8,
            'n_estimators': 500,
            'random_state': 42
        }

    logging.info(f"Training {commodity_name} model with parameters: {params}")
    model = xgb.XGBRegressor(**params)
    model.fit(train_X, train_y)

    y_pred = model.predict(test_X)

    mse = np.mean((test_y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_y - y_pred))

    logging.info(f"{commodity_name} Performance - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plot_predictions(test_y, y_pred, data['test_dates'], commodity_name)

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        },
        'scaler': data['scaler'],
        'feature_columns': data['exog_cols']
    }

def plot_predictions(test_y, predictions, dates, commodity_name):
    """Plot actual vs predicted prices for the last 6 months"""
    logging.info(f"Plotting predictions for {commodity_name}")
    if isinstance(test_y, pd.Series):
        pred_series = pd.Series(predictions, index=test_y.index)
    else:
        pred_series = pd.Series(predictions)

    if isinstance(dates, pd.Series) and len(dates) == len(test_y):
        last_6_months_start = dates.iloc[-1] - pd.DateOffset(months=6)
        mask = dates >= last_6_months_start

        plt.figure(figsize=(12, 6))
        plt.plot(dates[mask], test_y[mask], label='Actual Close', linewidth=2)
        plt.plot(dates[mask], pred_series[mask], label='Predicted Close', linewidth=2)
    else:
        test_y_plot = test_y.iloc[-180:] if len(test_y) > 180 else test_y
        pred_plot = pred_series.iloc[-180:] if len(pred_series) > 180 else pred_series

        plt.figure(figsize=(12, 6))
        plt.plot(test_y_plot, label='Actual Close', linewidth=2)
        plt.plot(pred_plot, label='Predicted Close', linewidth=2)

    plt.title(f'XGBoost: {commodity_name} Actual vs Predicted Close Price (Last 6 Months)')
    plt.xlabel('Date')
    plt.ylabel(f'{commodity_name} Close Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    models_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(models_dir, exist_ok=True)

    plot_path = os.path.join(models_dir, f"{commodity_name.lower().replace(' ', '_')}_prediction.png")
    plt.savefig(plot_path)
    logging.info(f"Saved plot to {plot_path}")
    plt.close()

def save_model(model_dict, commodity_name):
    """Save model and related data to disk"""
    models_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{commodity_name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model_dict, model_path)
    logging.info(f"Saved model to {model_path}")

def train_models():
    """Train models for all commodities"""
    logging.info("Starting model training for all commodities")
    data_dict = prepare_data()

    display_names = {
        'gold': 'Gold',
        'crude_oil': 'Crude Oil',
        'natural_gas': 'Natural Gas'
    }

    trained_models = {}
    for commodity_key, data in data_dict.items():
        display_name = display_names.get(commodity_key, commodity_key)
        model_dict = train_xgboost(data, display_name)
        save_model(model_dict, commodity_key)
        trained_models[commodity_key] = model_dict

    logging.info("Training complete for all commodities")
    return trained_models

if __name__ == "__main__":
    models = train_models()
    logging.info("Script execution finished")
