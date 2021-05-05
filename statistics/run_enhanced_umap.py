import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
import hdbscan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import sys
import argparse
def runModel():

    stocks = pd.read_csv(args.data_set, index_col = 0)

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

    # UMAP application
    standard_embedding = umap.UMAP(
        random_state=42,
    ).fit_transform(normalized_movements)

    clusterable_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(normalized_movements)

    #HDBSCAN
    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500,
    ).fit_predict(clusterable_embedding)

    clustered = (labels >= 0)

    # create two more arrays of resulting features to go into x and y coordinates of scatter plot
    xs = standard_embedding[~clustered,0]
    ys = standard_embedding[~clustered,1]

    #scatter plot
    fig, ax = plt.subplots(figsize = [15, 10])
    plt.scatter(xs, ys, alpha=0.5)

    for x, y, company in zip(xs, ys, companies):
        plt.annotate(company, (x, y), fontsize=9, alpha=0.75)
    plt.tight_layout
    plt.title('UMAP Projection of SP500 Stocks Dataset', fontsize=24)
    plt.savefig(args.output)

def help_statement():
    print(" ")
    print("Sample usage (from root): python statistics/run_enhanced_umap.py data/SP500_stock.csv output/enhanced_umap_model.png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_set', help='Dataset to be processed')
    parser.add_argument('output', help='Output destination')

    #Get and check options
    args = None
    
    #Check for argument flags
    if(len(sys.argv) == 1):
        help_statement()
        parser.print_help()
        sys.exit(0)
    elif(sys.argv[1] == '-h' or \
         sys.argv[1] == '--h' or \
         sys.argv[1] == '-help' or \
         sys.argv[1] == '--help'):
         help_statement()
         parser.print_help()
         sys.exit(0)
    else:
        args = parser.parse_args()
        print("Dataset: " + args.data_set)
        print("Output: " + args.output)
    
    #Try 
    try:
        runModel()
    except KeyboardInterrupt:
        print('Cancelled')
        raise
