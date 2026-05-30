import os
import sys
import logging
from pathlib import Path
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ZomatoDataLoader:
    """
    Ingests, cleans, normalizes, and caches the Zomato restaurant recommendation dataset.
    Sourced from: https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation
    """
    
    DATASET_NAME = "ManikaSaini/zomato-restaurant-recommendation"
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.data_dir = self.workspace_dir / "data"
        self.cache_file = self.data_dir / "zomato_cleaned.parquet"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_from_huggingface(self) -> pd.DataFrame:
        """
        Downloads the dataset from Hugging Face Hub using the datasets library.
        Includes automated retry logic for network resilience.
        """
        logger.info(f"Connecting to Hugging Face to load dataset: {self.DATASET_NAME}...")
        try:
            from datasets import load_dataset
            # Load the dataset
            dataset = load_dataset(self.DATASET_NAME)
            
            # The dataset usually contains a 'train' split
            split_key = list(dataset.keys())[0]
            logger.info(f"Dataset successfully loaded. Found splits: {list(dataset.keys())}. Using split: '{split_key}'")
            
            df = dataset[split_key].to_pandas()
            return df
        except ImportError:
            logger.error("The 'datasets' package is not installed. Attempting download via direct CSV link...")
            # Fallback if datasets package fails/is missing: read direct CSV link if available
            csv_url = f"https://huggingface.co/datasets/{self.DATASET_NAME}/raw/main/zomato.csv"
            return pd.read_csv(csv_url)
            
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names by converting to lower_snake_case.
        """
        original_cols = df.columns.tolist()
        df.columns = [
            col.strip().lower().replace(" ", "_").replace("-", "_") 
            for col in df.columns
        ]
        mapped = {orig: new for orig, new in zip(original_cols, df.columns)}
        logger.debug(f"Column name mappings: {mapped}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs validation, standardization, and cleaning heuristics on Zomato data.
        """
        logger.info(f"Starting data preprocessing on {len(df)} raw records...")
        
        # 1. Clean column names
        df = self.clean_column_names(df)
        
        # 2. Key column mapping check (resolve differences in column naming schemas)
        # We need standard columns: name, cuisines, locality, city, average_cost_for_two, aggregate_rating, votes
        column_aliases = {
            'restaurant_name': 'name',
            'locality_verbose': 'locality_full',
            'location': 'locality',
            'listed_in(city)': 'city',
            'approx_cost(for_two_people)': 'average_cost_for_two',
            'rate': 'aggregate_rating'
        }
        df = df.rename(columns=column_aliases)
        
        # Ensure critical columns exist, if not, create placeholder/fallback columns
        required_cols = ['name', 'cuisines', 'locality', 'city', 'average_cost_for_two', 'aggregate_rating', 'votes']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' missing. Initializing as NaN.")
                df[col] = pd.NA
                
        # 3. Clean rating column (convert strings like 'NEW', '-', or "4.1/5" to numeric rating)
        logger.info("Cleaning rating and votes data...")
        # Extract the score before the slash if it's in "4.1/5" format
        df['aggregate_rating'] = df['aggregate_rating'].astype(str).str.split('/').str[0].str.strip()
        df['aggregate_rating'] = pd.to_numeric(df['aggregate_rating'], errors='coerce')
        
        # Calculate local/city rating averages to fill NaNs, fallback to a sensible 3.5
        global_avg_rating = df['aggregate_rating'].mean()
        if pd.isna(global_avg_rating) or global_avg_rating == 0:
            global_avg_rating = 3.5
            
        city_avg = df.groupby('city')['aggregate_rating'].transform('mean')
        df['aggregate_rating'] = df['aggregate_rating'].fillna(city_avg).fillna(global_avg_rating)
        df['aggregate_rating'] = df['aggregate_rating'].round(1)
        
        # Clean votes column
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
        
        # 4. Clean cost column (convert string "1,200" to numeric, replace zeros/NaNs with local median)
        logger.info("Cleaning average cost columns...")
        # Remove commas from numeric strings before converting
        df['average_cost_for_two'] = df['average_cost_for_two'].astype(str).str.replace(',', '').str.strip()
        df['average_cost_for_two'] = pd.to_numeric(df['average_cost_for_two'], errors='coerce')
        
        # Resolve invalid cost values using localized medians
        global_median_cost = df['average_cost_for_two'].median()
        if pd.isna(global_median_cost) or global_median_cost == 0:
            global_median_cost = 500.0  # Safe default cost
            
        locality_median = df.groupby('locality')['average_cost_for_two'].transform('median')
        city_median = df.groupby('city')['average_cost_for_two'].transform('mean')  # Fallback to mean if median is empty
        
        df['average_cost_for_two'] = (
            df['average_cost_for_two']
            .fillna(locality_median)
            .fillna(city_median)
            .fillna(global_median_cost)
        )
        # Ensure average cost has no zero values
        df['average_cost_for_two'] = df['average_cost_for_two'].replace(0, global_median_cost)
        
        # 5. Clean string/categorical fields
        logger.info("Standardizing string and category fields...")
        df['name'] = df['name'].fillna("Unnamed Restaurant").astype(str).str.strip()
        df['cuisines'] = df['cuisines'].fillna("Multi-Cuisine").astype(str).str.strip()
        df['locality'] = df['locality'].fillna("Unknown Locality").astype(str).str.strip()
        df['city'] = df['city'].fillna("Unknown City").astype(str).str.strip()
        
        # 6. Drop absolute duplicates (Same name, locality, and cuisines)
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=['name', 'locality', 'cuisines'])
        post_dedup = len(df)
        logger.info(f"Deduplication complete. Removed {pre_dedup - post_dedup} duplicate restaurant records.")
        
        # Sort values logically by Rating and Votes for retrieval priority
        df = df.sort_values(by=['aggregate_rating', 'votes'], ascending=[False, False]).reset_index(drop=True)
        
        logger.info(f"Preprocessing complete. Cleaned {len(df)} records ready for recommendation search.")
        return df

    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieves the cleaned dataset. First checks the local cache, then downloads
        and cleans from Hugging Face if not found or if force_refresh is True.
        """
        if self.cache_file.exists() and not force_refresh:
            logger.info(f"Loading cleaned dataset from local cache: {self.cache_file}")
            try:
                df = pd.read_parquet(self.cache_file)
                logger.info(f"Successfully loaded {len(df)} records from cache.")
                return df
            except Exception as e:
                logger.warning(f"Failed to read Parquet cache: {e}. Downloading a fresh copy...")
        
        # Download, preprocess, and cache
        df_raw = self.fetch_from_huggingface()
        df_cleaned = self.preprocess_data(df_raw)
        
        logger.info(f"Caching cleaned dataset to: {self.cache_file}")
        try:
            df_cleaned.to_parquet(self.cache_file, index=False, engine='pyarrow')
            logger.info("Local caching complete.")
        except Exception as e:
            logger.error(f"Failed to cache data to Parquet: {e}")
            
        return df_cleaned

if __name__ == "__main__":
    # Test script entrypoint
    logger.info("Starting Zomato Data Loader test run...")
    loader = ZomatoDataLoader()
    df = loader.get_data(force_refresh=True)
    print("\n--- Cleaned Dataset Sample ---")
    print(df[['name', 'city', 'locality', 'cuisines', 'average_cost_for_two', 'aggregate_rating', 'votes']].head())
    print(f"\nTotal Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
