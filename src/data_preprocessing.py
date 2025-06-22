import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'data_preprocessing.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_raw_data():
    """Load raw commodity data from CSV file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', 'data', 'raw.csv')
    logging.info(f"Loading data from {filepath}")

    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

def filter_commodities(df):
    """Filter dataframe for target commodities and convert date"""
    logging.info("Filtering for Gold, Crude Oil, and Natural Gas")
    filtered_df = df[df['Commodity'].isin(['Gold', 'Crude Oil', 'Natural Gas'])].copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    logging.info(f"Filtered data shape: {filtered_df.shape}")
    return filtered_df

def split_by_commodity(df):
    """Split dataframe into separate dataframes for each commodity"""
    logging.info("Splitting data by commodity")
    commodities = {}
    for commodity in ['Gold', 'Crude Oil', 'Natural Gas']:
        commodity_df = df[df['Commodity'] == commodity].copy()
        commodity_df.sort_values('Date', inplace=True)
        commodity_df.reset_index(drop=True, inplace=True)
        columns_to_drop = ['Commodity', 'Open', 'High', 'Low']
        commodity_df.drop(columns=columns_to_drop, inplace=True)
        commodity_key = commodity.replace(' ', '_').lower()
        commodities[commodity_key] = commodity_df
        logging.info(f"{commodity_key} data prepared with shape {commodity_df.shape}")
    return commodities

def preprocessing_xgboost(df, test_size=0.2, n_lags=30):
    """Preprocess data for XGBoost training"""
    logging.info("Starting preprocessing for XGBoost")
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    df['Close_roll_mean_7'] = df['Close'].rolling(window=7).mean()
    df['Close_roll_std_7'] = df['Close'].rolling(window=7).std()
    df['Close_target'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    logging.info(f"Data after feature engineering: {df.shape}")
    exog_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Close_target']]
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    scaler = StandardScaler()
    train_X = pd.DataFrame(scaler.fit_transform(train_df[exog_cols]), columns=exog_cols)
    test_X = pd.DataFrame(scaler.transform(test_df[exog_cols]), columns=exog_cols)
    train_y = train_df['Close_target']
    test_y = test_df['Close_target']
    logging.info(f"Train/Test split: {train_X.shape[0]} train, {test_X.shape[0]} test")
    return {
        'train_X': train_X,
        'test_X': test_X,
        'train_y': train_y,
        'test_y': test_y,
        'scaler': scaler,
        'exog_cols': exog_cols,
        'train_dates': train_df['Date'],
        'test_dates': test_df['Date']
    }

def prepare_data():
    """Main function to prepare data for all commodities"""
    logging.info("Starting data preparation pipeline")
    raw_data = load_raw_data()
    filtered_data = filter_commodities(raw_data)
    commodity_dfs = split_by_commodity(filtered_data)
    processed_data = {}
    for commodity_name, df in commodity_dfs.items():
        processed_data[commodity_name] = preprocessing_xgboost(df)
        logging.info(f"Processed {commodity_name}: {len(df)} rows")
    return processed_data

if __name__ == "__main__":
    logging.info("Running data preprocessing script")
    data = prepare_data()
    logging.info("Data preprocessing complete!")
    for commodity, dataset in data.items():
        logging.info(f"{commodity}: {dataset['train_X'].shape[0]} training samples, {dataset['test_X'].shape[0]} testing samples")
