import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


stocks = pd.read_csv('data/SP500_stock.csv', index_col = 0)

stocks['stock_change'] = stocks['close'] - stocks['open'] # Create new column of difference between close and open values
stocks = stocks.drop(['open','high','low','close','volume'], axis=1) # Drop unnecessary columns
stocks = stocks.reset_index() 
stocks = stocks.pivot(index='Name', columns='date') # Restructure dataframe

stocks = stocks.fillna(0) #pandas version of replacing NaN values to 0
#np.nan_to_num(stocks) # numpy version of replacing NaN values to 0

#print(stocks.head(5)) #test stocks dataframe by printing first 5 rows

movements = stocks.values
companies = stocks.index

normalized_movements = normalize(movements)

model = TSNE(learning_rate = 50)

tsne_features = model.fit_transform(normalized_movements)

xs = tsne_features[:,0]
ys = tsne_features[:,1]

fig, ax = plt.subplots(figsize = [15, 10])
plt.scatter(xs, ys, alpha = 0.5)

for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=9, alpha=0.75)
plt.tight_layout
plt.savefig('output/model.png') #temporary fix, this saves output as a png file
