from src.preprocessing import CSVLoader



if __name__ == "__main__":
    loader = CSVLoader()
    df_true = loader.load_csv("data/row/True.csv")



    