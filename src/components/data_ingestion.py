import os

import pandas as pd


def load_data(

    data_path="data"

):

    train_path = os.path.join(

        data_path,

        "de_train.parquet"

    )

    id_map_path = os.path.join(

        data_path,

        "id_map.csv"

    )

    df = pd.read_parquet(

        train_path

    )

    id_map = pd.read_csv(

        id_map_path

    )

    return df, id_map