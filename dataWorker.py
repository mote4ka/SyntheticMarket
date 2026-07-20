import requests
import pandas as pd

def getOHLC(ticker:str, start: int, end:int, normalize:bool = False) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"

    params = {
        "category": "linear", 
        "symbol": ticker,
        "interval": "1",    
        "start": start,
        "end": end,
        "limit": 1000
    }
 
    req = requests.get(url, params)

    data = req.json().get('result').get('list')

    columns = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.DataFrame(data, columns=columns)

    # changes values to float
    num_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    df[num_cols] = df[num_cols].astype(float)

    # timestamp -> datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['volume'] = 1
    
    if normalize:
        # normallize prices
        base_price = df.loc[0]['open']
        df[['open', 'high', 'close', 'low']] = (df[['open', 'high', 'close', 'low']] - base_price) / base_price * 100
    
    return df

import random

def getRandomIntervalData(ticker:str, duration:int=120, normalize:bool = False) -> pd.DataFrame:
    random_hour = random.randint(0,10000)
    start = ( 1784365200-60*60*random_hour-60*duration )*1000
    end = ( 1784365200-60*60*random_hour )*1000
    
    df = getOHLC(ticker, start, end, normalize=normalize)
    
    return start,end,duration,df
