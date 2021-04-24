# professionalstudioB
Data Engineering - Professional Studio B Project Repository

Dataset used: https://www.kaggle.com/camnugent/sandp500

** Required dependencies **  
pip install numpy  
pip install pandas  
pip install umap-learn  
pip install argparse  

(There may be some modules that I forgot, but normally executing pip install will get them)

# t-SNE Model
Source article: https://medium.com/analytics-vidhya/machine-learning-for-the-stock-market-use-python-to-find-companies-that-behave-similarly-81eceee04f2c

**Sample Usage (from root directory)**  
-> python tsne/run.py data/SP500_stock.csv output/tsne_model.png

**Sample Result**  
![Alt text](/output/tsne_model.jpg?raw=true)

