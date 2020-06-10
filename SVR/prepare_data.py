import os
from pathlib import Path
from pandas_datareader import data as pdr

import yfinance as yf

yf.pdr_override()  # <== that's all it takes :-)

START_DATE_TRAIN = "2010-08-01"
END_DATE_TRAIN = "2019-07-06"
START_DATE_COMPARISION = "2019-07-07"
END_DATE_COMPARISION = "2019-07-17"
START_DATE_PREDICTION = "2019-07-18"
END_DATE_PREDICTION = "2019-07-24"
tickers = ['SPY', 'BABA', 'AAPL']


# download dataframe using pandas_datareader
def build_stock_dataset_for_train(start=START_DATE_TRAIN, end=END_DATE_TRAIN):
    Path("./data").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start, end) # Date,Open,High,Low,Close,Volume
        data.to_csv("./data/" + ticker + ".csv")


def build_stock_dataset_for_comparision(start=START_DATE_COMPARISION, end=END_DATE_COMPARISION):
    Path("./dataComparision").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start, end) # Date,Open,High,Low,Close,Volume
        data.to_csv("./dataComparision/" + ticker + ".csv")

def build_stock_dataset_for_prediction(start=START_DATE_PREDICTION, end=END_DATE_PREDICTION):
    Path("./dataPrediction").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start, end) # Date,Open,High,Low,Close,Volume
        data.to_csv("./dataPrediction/" + ticker + ".csv")


if __name__ == "__main__":
    build_stock_dataset_for_train()
    build_stock_dataset_for_comparision()