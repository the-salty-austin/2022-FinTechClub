import requests

from datetime import datetime
import pandas as pd
import csv
from utility import timer
import time

url = "https://www.binance.com/api/v3/klines"


@timer
def get_data_since(symbol: str, startTime: datetime, endTime: datetime, interval: str) -> pd.DataFrame:
    """
    symbol: BTC, ETH, BNB... \n
    startTime: when to start \n
    endTime: last row of output is no earlier than it
    interval: '1m', '1h', '4h', '1d'
    """
    data = []

    unixTimeNow = startTime.timestamp() * 1000

    while unixTimeNow <= endTime.timestamp() * 1000:
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": "1000",
            "startTime": int(unixTimeNow),
        }
        price = requests.get(url, params=params).json()
        # pprint.pprint( price )
        """
        Response Format
            [
                [
                    1499040000000,      // Kline open time
                    "0.01634790",       // Open price
                    "0.80000000",       // High price
                    "0.01575800",       // Low price
                    "0.01577100",       // Close price
                    "148976.11427815",  // Volume
                    1499644799999,      // Kline Close time
                    "2434.19055334",    // Quote asset volume
                    308,                // Number of trades
                    "1756.87402397",    // Taker buy base asset volume
                    "28.46694368",      // Taker buy quote asset volume
                    "0"                 // Unused field, ignore.
                ]
            ]
        """
        for row in price:
            # time, open, high, low, close, volume
            t, o, h, l, c, v = (
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
            unixTimeNow = t
            data.append([t, o, h, l, c, v])
            # print( [t, o, h, l, c, v] )。
        if interval=="1m": 
            s = 60*1*1
        elif interval=="1h":
            s = 60*60*1
        elif interval=="4h":
            s = 60*60*4
        elif interval=="1d":
            s = 60*60*24
        unixTimeNow += s*1000
        time.sleep(0.5)

    df = pd.DataFrame(
        data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df.timestamp = pd.to_datetime(df.timestamp, unit="ms")

    return df


if __name__ == "__main__":
    df = get_data_since(
        "BTC",
        datetime(2021, 1, 1, 0, 0),
        datetime(2022, 12, 24, 12, 30),
        "1h"
    )
    df.to_csv('./csv/btc1h.csv', index=False)
    print(df)
