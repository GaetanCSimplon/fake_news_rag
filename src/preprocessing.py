import pandas as pd
from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load_csv(self, path) -> pd.DataFrame:
        """Abstract method to load a CSV file."""
        pass


class CSVLoader(DataLoader):
    def load_csv(self, path) -> pd.DataFrame:
        print("Start loading ...!")
        self.path = path
        self.df = pd.read_csv(self.path)
        print(self.df)
        return self.df
