import os
import pandas as pd
from z_score import make_z_score_data
from pandas import DataFrame


def find_data():
    Dir = r"input datapath"
    filename = ["train", "test", "val"]

    label = 2
    cc = 13  # 0：P_ID 1:Label 2:f1~fn
    Epoch = 250

    return Dir, filename, label, cc, Epoch



def get_data(Dir, filename, label, cc):
    filename = filename+".csv"
    data = pd.read_csv(os.path.join(Dir, filename))
    x_data = data.iloc[:, cc:]
    y_data = data.iloc[:, label-1]
    ID_data = data.iloc[:, label - 2]
    return x_data, y_data, ID_data
