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
-> python statistics/run_tsne.py data/SP500_stock.csv output/tsne_model.png

# PCA Model  
**Sample Result**
![image](https://github.com/filicasc/professionalstudioB/blob/main/output/pca_model.png?raw=true)

**Sample Result**  
![image](https://github.com/filicasc/professionalstudioB/blob/main/output/tsne_model.png?raw=true)

# UMAP Model
UMAP documentation: https://umap-learn.readthedocs.io/en/latest/index.html

**Sample Usage (from root directory)**  
-> python statistics/run_umap.py data/SP500_stock.csv output/umap_model.png

**Sample Results**
![image](https://github.com/filicasc/professionalstudioB/blob/main/output/umap_model.png?raw=true)  
![image](https://github.com/filicasc/professionalstudioB/blob/main/output/umap_model_test.png?raw=true)

# Enhanced UMAP Model (with HDBSCAN)  
This model includes the application of HDBSCAN, a density based clustering algorithm. Additional parameters have been included for UMAP to get the dataset down to a number of dimensions small enough for a density based clustering algorithm to make progress.  
Source: https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering  

**Sample Usage (from root directory)**  
-> python statistics/run_enhanced_umap.py data/SP500_stock.csv output/enhanced_umap_model.png

**Sample Result**
![image](https://github.com/filicasc/professionalstudioB/blob/main/output/enhanced_umap_model.png?raw=true)



