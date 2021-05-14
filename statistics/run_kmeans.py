import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import manifold
from sklearn.cluster import KMeans

stocks = pd.read_csv('data/SP500_stock.csv', index_col = 0)

stocks['stock_change'] = stocks['close'] - stocks['open'] # Create new column of difference between close and open values
stocks = stocks.drop(['open','high','low','close','volume'], axis=1) # Drop unnecessary columns
stocks = stocks.reset_index() 
stocks = stocks.pivot(index='Name', columns='date') # Restructure dataframe

stocks = stocks.fillna(0) #pandas version of replacing NaN values to 0

# create two arrays of values and index (companies)
movements = stocks.values
companies = stocks.index

# sklearn's normalize function to convert values into the same scale
normalized_movements = normalize(movements)

# K-Means
kmeans = KMeans(n_clusters = 5)
transformed = kmeans.fit_transform(normalized_movements)

# create two more arrays of resulting features to go into x and y coordinates of scatter plot
xs = transformed[:,0]
ys = transformed[:,1]

# create scatter plot
fig, ax = plt.subplots(figsize = [15, 10])
plt.scatter(xs, ys, alpha = 0.5)

temp_list=[]

for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=9, alpha=0.75)
    temp_list.append([company,x,y])
plt.tight_layout
plt.title('Kmeans Projection of SP500 Stocks Dataset', fontsize=24)
plt.savefig('output/kmeans_model.png') #temporary fix, this saves output as a png file

df = pd.DataFrame(temp_list, columns=['Company','X','y'])
df.to_csv("output/kmeans_file.csv", sep=',', index=False)

