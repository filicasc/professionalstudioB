import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


stocks = pd.read_csv('data/SP500_stock.csv', index_col = 0)
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
plt.savefig('test.png') #temporary fix, this saves output as a png file