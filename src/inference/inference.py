import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging for inference
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'inference.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing import prepare_data

def load_model(commodity_name):
    """Load trained model for a specific commodity"""
    models_dir = os.path.join(os.path.dirname(__file__), '../../models')
    model_path = os.path.join(models_dir, f"{commodity_name}_model.pkl")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logging.info(f"Loaded model for {commodity_name} from {model_path}")
    return joblib.load(model_path)

def evaluate_model(model_dict, test_X, test_y):
    """Evaluate model performance on test data"""
    model = model_dict['model']
    predictions = model.predict(test_X)
    
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    
    logging.info(f"Evaluation metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {
        'predictions': predictions,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }

def plot_metrics_table(metrics_dict, commodities):
    """Create and save a table of evaluation metrics for all commodities"""
    data = {
        'Commodity': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R²': []
    }
    
    for commodity in commodities:
        data['Commodity'].append(commodity.replace('_', ' ').title())
        data['MSE'].append(f"{metrics_dict[commodity]['mse']:.4f}")
        data['RMSE'].append(f"{metrics_dict[commodity]['rmse']:.4f}")
        data['MAE'].append(f"{metrics_dict[commodity]['mae']:.4f}")
        data['R²'].append(f"{metrics_dict[commodity]['r2']:.4f}")
    
    metrics_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    plt.title('Commodity Price Prediction - Model Evaluation Metrics', fontsize=16, pad=20)
    
    performance_dir = os.path.join(os.path.dirname(__file__), '../../performance')
    os.makedirs(performance_dir, exist_ok=True)
    
    save_path = os.path.join(performance_dir, 'evaluation_metrics_table.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Saved evaluation metrics table plot to {save_path}")

def save_metrics_to_csv(metrics_dict, commodities):
    """Save evaluation metrics to a CSV file"""
    data = {
        'Commodity': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R2': []
    }
    
    for commodity in commodities:
        data['Commodity'].append(commodity.replace('_', ' ').title())
        data['MSE'].append(metrics_dict[commodity]['mse'])
        data['RMSE'].append(metrics_dict[commodity]['rmse'])
        data['MAE'].append(metrics_dict[commodity]['mae'])
        data['R2'].append(metrics_dict[commodity]['r2'])
    
    metrics_df = pd.DataFrame(data)
    performance_dir = os.path.join(os.path.dirname(__file__), '../../performance')
    os.makedirs(performance_dir, exist_ok=True)
    
    csv_path = os.path.join(performance_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    logging.info(f"Saved evaluation metrics CSV to {csv_path}")

def plot_test_predictions(test_y, predictions, dates, commodity_name):
    """Plot actual vs predicted prices for test data"""
    logging.info(f"Plotting test predictions for {commodity_name}")
    plt.figure(figsize=(12, 6))
    
    if isinstance(test_y, pd.Series):
        pred_series = pd.Series(predictions, index=test_y.index)
    else:
        pred_series = pd.Series(predictions)
    
    if isinstance(dates, pd.Series) and len(dates) == len(test_y):
        plt.plot(dates, test_y, label='Actual Close', linewidth=2, color='blue')
        plt.plot(dates, pred_series, label='Predicted Close', linewidth=2, color='red')
        plt.xlabel('Date')
    else:
        plt.plot(test_y.values, label='Actual Close', linewidth=2, color='blue')
        plt.plot(pred_series, label='Predicted Close', linewidth=2, color='red')
        plt.xlabel('Test Sample Index')
    
    plt.title(f'{commodity_name.replace("_", " ").title()} - Test Data: Actual vs Predicted Close Prices')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    performance_dir = os.path.join(os.path.dirname(__file__), '../../performance')
    os.makedirs(performance_dir, exist_ok=True)
    
    plot_path = os.path.join(performance_dir, f'{commodity_name}_test_predictions.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved test predictions plot for {commodity_name} to {plot_path}")

def run_inference():
    logging.info("Starting inference run")
    data_dict = prepare_data()
    
    commodities = ['gold', 'crude_oil', 'natural_gas']
    all_metrics = {}
    
    for commodity in commodities:
        logging.info(f"Evaluating model for {commodity}")
        print(f"\nEvaluating {commodity.replace('_', ' ').title()} model...")
        
        model_dict = load_model(commodity)
        test_X = data_dict[commodity]['test_X']
        test_y = data_dict[commodity]['test_y']
        test_dates = data_dict[commodity]['test_dates']
        
        evaluation = evaluate_model(model_dict, test_X, test_y)
        metrics = evaluation['metrics']
        
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        
        all_metrics[commodity] = metrics
        
        plot_test_predictions(test_y, evaluation['predictions'], test_dates, commodity)
    
    plot_metrics_table(all_metrics, commodities)
    save_metrics_to_csv(all_metrics, commodities)
    logging.info("Inference run complete")
    print("\nEvaluation complete! Results saved to 'performance' directory.")

if __name__ == "__main__":
    run_inference()
