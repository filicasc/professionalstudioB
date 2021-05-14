# professionalstudioB
Data Engineering - Professional Studio B Project Repository

Dataset used: https://www.kaggle.com/camnugent/sandp500  

# Dependencies  
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

**Sample Result**  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/tsne_model.png?raw=true" width="750" height="500">  

# PCA Model  
**Sample Result**  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/pca_model.png?raw=true" width="750" height="500">  

# K-means Clustering Model  
**Sample Result**  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/kmeans_model.png?raw=true" width="750" height="500">  

# UMAP Model
UMAP documentation: https://umap-learn.readthedocs.io/en/latest/index.html

**Sample Usage (from root directory)**  
-> python statistics/run_umap.py data/SP500_stock.csv output/umap_model.png

**Sample Results**  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/umap_model.png?raw=true" width="750" height="500">  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/umap_model_test.png?raw=true" width="750" height="500">  

# Enhanced UMAP Model (with HDBSCAN)  
This model includes the application of HDBSCAN, a density based clustering algorithm. Additional parameters have been included for UMAP to get the dataset down to a number of dimensions small enough for a density based clustering algorithm to make progress.  
Source: https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering  

**Sample Usage (from root directory)**  
-> python statistics/run_enhanced_umap.py data/SP500_stock.csv output/enhanced_umap_model.png

**Sample Result**  
<img src="https://github.com/filicasc/professionalstudioB/blob/main/output/enhanced_umap_model.png?raw=true" width="750" height="500">  

# Clustering Performance Evaluation  
We looked at numerous methods to evaluate the performance of the tested clustering methods. Considering our scenario, we went with the Calinski-Harabasz Index.  
Source: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation  

If the ground truth labels are not known, this index (also known as Variance Ratio Criterion) can be used to evaluate clustering models, where a higher value indicates a model with better defined clusters. The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared).  

| Cluster Algorithm | Calinski-Harabasz Index |
| ----------------- |:-----------------------:|
| UMAP & HDBSCAN    | 1797.491                |
| UMAP              | 1347.602                |
| t-SNE             | 780.334                 |
| PCA               | 685.493                 |
| K-means           | 584.692                 |  
