# pustaka generate number
import random as rm
import time as tm

# pustaka manipulasi data array
import numpy as np

# pustaka manipulasi data frame
import pandas as pd


def loadData(df):
    dataset = pd.read_csv("D:/2024_BOPTN_Hopsot/source-code/temp/"+df)
    return dataset