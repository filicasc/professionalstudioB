import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import argparse

stocks = pd.read_csv('../data/SP500_stock.csv', index_col = 0)

stocks['stock_change'] = stocks['close'] - stocks['open'] # Create new column of difference between close and open values
stocks = stocks.drop(['open','high','low','close','volume'], axis=1) # Drop unnecessary columns
stocks = stocks.reset_index() 
stocks = stocks.pivot(index='Name', columns='date') # Restructure dataframe

stocks = stocks.fillna(0) #pandas version of replacing NaN values to 0

#print(stocks.head(5))

reducer = umap.UMAP()