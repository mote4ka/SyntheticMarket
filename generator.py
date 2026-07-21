import pandas as pd
import numpy as np
import random



def distribution(sigma=0.2, mode="gauss", df=3, high=None):
    
    if mode == "uniform":
        x = random.uniform(-sigma, sigma)
    elif mode == "gauss":
        x = random.gauss(0, sigma)
    elif mode == "student_t":
        x = np.random.standard_t(df) * sigma
        
    if high is not None:
        x = max(min(x, high), -high)
    return x


def generateSyntheticChart(minutes:int = 125, seed=-1):
    if seed == -1:
        seed = int(np.random.default_rng().integers(0,2**23))

    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # parametes
    body_sigma = 0.046
    wick_sigma = 0.02
    body_df = 4
    wick_df = 5

    data = [0]*minutes
    base_price = 0

    for i in range(minutes):
        timestamp = (1784206800 + 60*i)*1000
        if i==0: 
            price_o = base_price
        else: 
            price_o = data[i-1][4]
            
        price_c = price_o + distribution(sigma=body_sigma, mode="gauss", df=body_df)
        price_h = max(price_o, price_c) + abs(distribution(sigma=wick_sigma, mode="student_t", df=wick_df))
        price_l = min(price_o, price_c) - abs(distribution(sigma=wick_sigma, mode="student_t", df=wick_df))
            
            
        #              0         1        2        3        4   
        data[i] = [timestamp, price_o, price_h, price_l, price_c]

    columns = ["timestamp", "open", "high", "low", "close"]
    df = pd.DataFrame(data, columns=columns)
    
    return df, seed

import os

# saving list of pd.DataFrames to folder
# input dstet -> [ pd.DataFrame, seed ]
def saveDataSet(dtset, dir):
    os.makedirs(dir, exist_ok=True)
    
    for i in range(len(dtset)):
        path = f"{dir}/seed_{dtset[i][1]}.parquet"
        dtset[i][0].to_parquet(path, index=False)
    
from pathlib import Path

def loadDataSet(dir="data/synthetic/train"):
    paths = sorted(Path(dir).glob("*.parquet"))
    dfs = [pd.read_parquet(p) for p in paths]
    return dfs


    