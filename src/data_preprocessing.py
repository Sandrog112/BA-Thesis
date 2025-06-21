import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Configure Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)  

log_file = os.path.join(LOG_DIR, f"data_preprocessing_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  
        logging.StreamHandler()        
    ]
)
logger = logging.getLogger()

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "../data"))
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw.csv")  
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed") 

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class CommodityPreprocessor:
    def __init__(self):
        self.df = None
        self.scaler = None

    def load_raw_data(self, raw_path: str) -> pd.DataFrame:
        """
        Loads the raw CSV data for commodities.
        """
        logger.info(f"Loading raw data from {raw_path}...")
        try:
            self.df = pd.read_csv(raw_path)
        except FileNotFoundError:
            logger.error(f"File not found: {raw_path}. Please ensure the path is correct.")
            raise
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise
        return self.df

    def filter_commodities(self, df: pd.DataFrame, commodities=['Gold', 'Crude Oil', 'Natural Gas']) -> pd.DataFrame:
        """
        Filters the dataframe to retain data for selected commodities.
        """
        logger.info(f"Filtering commodities: {commodities}")
        return df[df['Commodity'].isin(commodities)].reset_index(drop=True)

    def preprocess(self, df: pd.DataFrame, commodity: str, n_lags=3, save_path: str = None) -> pd.DataFrame:
        """
        Preprocesses data for the given commodity and saves the preprocessed data if `save_path` is provided.
        """
        logger.info(f"Preprocessing commodity data: {commodity}")

        df_commodity = df[df['Commodity'] == commodity].copy()
        df_commodity['Date'] = pd.to_datetime(df_commodity['Date'])
        df_commodity.sort_values('Date', inplace=True)
        df_commodity.reset_index(drop=True, inplace=True)

        columns_to_drop = ['Commodity', 'Open', 'High', 'Low']
        df_commodity.drop(columns=columns_to_drop, inplace=True)

        for lag in range(1, n_lags + 1):
            df_commodity[f'Close_lag_{lag}'] = df_commodity['Close'].shift(lag)

        df_commodity['Close_roll_mean_7'] = df_commodity['Close'].rolling(window=7).mean()
        df_commodity['Close_roll_std_7'] = df_commodity['Close'].rolling(window=7).std()

        df_commodity['Close_target'] = df_commodity['Close'].shift(-1)

        df_commodity.dropna(inplace=True)
        df_commodity.reset_index(drop=True, inplace=True)

        exog_cols = [col for col in df_commodity.columns if col not in ['Date', 'Close', 'Close_target']]
        self.scaler = StandardScaler()
        df_commodity[exog_cols] = self.scaler.fit_transform(df_commodity[exog_cols])

        # Save preprocessed data
        if save_path:
            self.save(df_commodity, save_path)

        logger.info(f"Preprocessing for {commodity} completed successfully.")
        return df_commodity

    def save(self, df: pd.DataFrame, out_path: str):
        """
        Saves the given dataframe to the specified path.
        """
        logger.info(f"Saving preprocessed data to: {out_path}")
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save data to {out_path}: {e}")
            raise
        logger.info(f"Data successfully saved to {out_path}")


# Main function for execution
if __name__ == "__main__":
    logger.info("Starting data preprocessing script...")

    try:
        preprocessor = CommodityPreprocessor()

        raw_df = preprocessor.load_raw_data(RAW_DATA_PATH)

        commodities = ['Gold', 'Crude Oil', 'Natural Gas']
        filtered_df = preprocessor.filter_commodities(raw_df, commodities)

        for commodity in commodities:
            save_path = os.path.join(PREPROCESSED_DIR, f"{commodity.lower().replace(' ', '_')}_preprocessed.csv")
            preprocessed_df = preprocessor.preprocess(filtered_df, commodity=commodity, save_path=save_path)

            logger.info(f"Processed and saved data for {commodity}: {preprocessed_df.shape[0]} rows, {preprocessed_df.shape[1]} columns.")

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")