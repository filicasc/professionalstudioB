import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import sys
import argparse

def runModel():
    stocks = pd.read_csv(args.data_set, index_col = 0)

    stocks['stock_change'] = stocks['close'] - stocks['open'] # Create new column of difference between close and open values
    stocks = stocks.drop(['open','high','low','close','volume'], axis=1) # Drop unnecessary columns
    stocks = stocks.reset_index() 
    stocks = stocks.pivot(index='Name', columns='date') # Restructure dataframe

    stocks = stocks.fillna(0) #pandas version of replacing NaN values to 0
    #np.nan_to_num(stocks) # numpy version of replacing NaN values to 0

    #print(stocks.head(5)) #test stocks dataframe by printing first 5 rows

    # create two arrays of values and index (companies)
    movements = stocks.values
    companies = stocks.index

    # sklearn's normalize function to convert values into the same scale
    normalized_movements = normalize(movements)

    # start an instance of t-SNE, learning rate should be adjusted depending on dataset (50-200)
    model = TSNE(learning_rate = 50)

    # apply t-SNE model to normalised array of stock prices
    tsne_features = model.fit_transform(normalized_movements)

    # create two more arrays of resulting features to go into x and y coordinates of scatter plot
    xs = tsne_features[:,0]
    ys = tsne_features[:,1]

    # create scatter plot
    fig, ax = plt.subplots(figsize = [15, 10])
    plt.scatter(xs, ys, alpha = 0.5)

    for x, y, company in zip(xs, ys, companies):
        plt.annotate(company, (x, y), fontsize=9, alpha=0.75)
    plt.tight_layout
    plt.title('t-SNE Projection of SP500 Stocks Dataset', fontsize=24)
    plt.savefig(args.output) #temporary fix, this saves output as a png file

def help_statement():
    print(" ")
    print("Sample usage (from root): python tsne/run.py data/SP500_stock.csv output/tsne_model.png")

if __name__ == '__main__':

    #project_root = dirname(dirname(__file__))
    #output = join(ROOT_DIR, 'output/tsne_model.png') # set output destination

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
