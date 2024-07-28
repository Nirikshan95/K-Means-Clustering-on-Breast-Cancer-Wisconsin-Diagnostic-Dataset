# K-Means Clustering on Breast Cancer Wisconsin (Diagnostic) Dataset

## Description
In this project, I implemented the K-Means clustering algorithm to identify clusters in the Breast Cancer Wisconsin (Diagnostic) dataset. The project demonstrates how to apply Principal Component Analysis (PCA) for dimensionality reduction and visualize the clustering results.

## Files Included
- `kmeans_clustering.ipynb`: Jupyter Notebook with the complete implementation.
- `README.md`: Explanation of the project.
- `requirements.txt`: Required Python libraries.
- `LICENSE`: Licensing information.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. It contains features computed from digitized images of fine needle aspirate (FNA) of breast mass. The dataset is available on the UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

## Project Structure
1. **Data Loading and Preprocessing**
   - Load the dataset and perform basic preprocessing.
   - Handle missing values and standardize the data.

2. **Dimensionality Reduction using PCA**
   - Apply PCA to reduce the dimensionality of the data for better visualization and computational efficiency.

3. **K-Means Clustering**
   - Implement the K-Means clustering algorithm.
   - Determine the optimal number of clusters using the Elbow method.

4. **Visualization**
   - Visualize the clustering results using scatter plots of the principal components.
   - Compare the clustering results with the true labels (if available).

## Usage
To run the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kmeans-clustering-breast-cancer.git
   cd kmeans-clustering-breast-cancer
   ```
2. Install the required libraries:
```
pip install -r requirements.txt
```
Open the Jupyter Notebook:
```
jupyter notebook kmeans_clustering.ipynb
```
## Acknowledgements
- UCI Machine Learning Repository for providing the Breast Cancer Wisconsin (Diagnostic) dataset.
- Scikit-learn for the machine learning tools and algorithms.
