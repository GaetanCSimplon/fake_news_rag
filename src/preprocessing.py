import pandas as pd
from abc import ABC, abstractmethod
import re
import os
from typing import List


class DataLoader(ABC):
    """
    Abstract base class for data loading.
    Any subclass must implement the load_csv method.
    """
    @abstractmethod
    def load_csv(self, path) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        pass


class CSVLoader(DataLoader):
    """
    Concrete implementation of DataLoader for CSV files.
    """
    def load_csv(self, path) -> pd.DataFrame:
        """
        Load a CSV file using pandas and return the DataFrame.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        print(f"[INFO] Start loading CSV from: {path}")
        self.path = path
        self.df = pd.read_csv(self.path)
        print(f"[INFO] CSV loaded successfully! Shape: {self.df.shape}")
        print(f"[DEBUG] Columns: {self.df.columns.tolist()}")
        return self.df

class DataCleaner:
    """
    Class to clean a pandas DataFrame.
    All cleaning methods operate directly on self.df.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
        """
        self.df = df
        print(f"[INFO] DataCleaner initialized with DataFrame shape: {self.df.shape}")
    
    # Labelling
    def add_label(self, label: int):
        """
        Add column "label" where 0 = fake 1 = True
        """
        self.df["label"] = label
        print(f"[INFO] Label '{label}' added to {len(self.df)} lines.")
        return self

    def drop_empty_rows_and_duplicated(self):
        """
        Drop rows with missing values in 'text', 'subject', or 'date' columns,
        and drop duplicate rows based on 'text'.
        """
        before_shape = self.df.shape
        self.df = self.df.dropna(subset=['text', 'subject', 'date'])
        self.df = self.df.drop_duplicates(subset=['text'], keep="first")
        after_shape = self.df.shape
        print(f"[INFO] drop_empty_rows_and_duplicated: before {before_shape}, after {after_shape}")
        return self

    def remove_spaces(self):
        """
        Remove leading and trailing spaces from all string columns.
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip()
        print(f"[INFO] remove_spaces: All string columns stripped of leading/trailing spaces")
        return self

    def lower_case(self):
        """
        Convert the 'text' column to lowercase.
        """
        if 'text' in self.df.columns:
            self.df['text'] = self.df['text'].str.lower()
            print(f"[INFO] lower_case: Converted 'text' column to lowercase")
        return self

    def date_format(self):
        """
        Convert the 'date' column to datetime format.
        Rows with invalid dates will be dropped.
        """
        if 'date' in self.df.columns:
            before_shape = self.df.shape
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            after_shape = self.df.shape
            print(f"[INFO] date_format: before {before_shape}, after {after_shape}")
        return self

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a string by removing extra spaces, URLs, and non-alphanumeric characters.
        """
        text = re.sub(r'\s+', ' ', text)          
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def clean_all_text_columns(self):
        """
        Apply the clean_text method to all string columns in the DataFrame.
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                print(f"[DEBUG] Cleaning column: {col}")
                self.df[col] = self.df[col].astype(str).apply(self.clean_text)
        print(f"[INFO] clean_all_text_columns: All string columns cleaned")
        return self
    
    def save_csv(self, path: str, index=False):
        """
        Save the current DataFrame to a CSV file.
        Will not overwrite if file already exists.

        Args:
            path (str): File path to save the CSV.
            index (bool): Whether to write row indices. Default is False.
        """
        if os.path.exists(path):
            print(f"[WARNING] File already exists: {path}. Skipping save.")
        else:
            self.df.to_csv(path, index=index)
            print(f"[INFO] DataFrame saved to: {path}")
        return self
    
    def get_df(self):
        """
        Return the cleaned DataFrame.
        """
        print(f"cleand data : \n{self.df}")
        return self.df
    
    
class DatasetMerger:
    """
    Merge multiple dataframes.
    """
    @staticmethod
    def merge(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Merge done.")
        return combined