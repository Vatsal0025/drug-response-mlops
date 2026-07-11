import os
import pandas as pd


def load_data(data_path):

    train_path = os.path.join(data_path, "de_train.parquet")
    test_path = os.path.join(data_path, "id_map.csv")

    df = pd.read_parquet(train_path)
    id_map = pd.read_csv(test_path)

    return df, id_map