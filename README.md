# Research Paper Recommendation System

This repository implements a Machine Learning-based recommendation system for research papers, leveraging metadata from the arXiv dataset. The project applies text preprocessing, clustering, and dimensionality reduction techniques to build a scalable and effective recommendation system.

## Dataset Overview

**Arxiv Dataset**: [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

The arXiv dataset is a collection of scientific papers submitted to arXiv.org, a repository of electronic preprints spanning various fields including mathematics, physics, computer science, and more. It is widely used by researchers for accessing the latest advancements in their domains. This dataset is a valuable resource for tasks like natural language processing, machine learning, and data mining.

### Dataset Features
- **ID**: Unique identifier for each paper.
- **Title**: Title of the paper.
- **Abstract**: Summary of the paper's content.
- **Authors**: List of authors.
- **Categories**: Classification of the paper under research domains.
- **Year**: Year of publication.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Insights into the distribution of papers by year, abstract length, and categories.
- Visualization of word clouds for top categories.

### 2. Preprocessing
- Text normalization: lowercasing, removing punctuation.
- Stopword removal and lemmatization.
- Feature extraction using TF-IDF vectorization.

### 3. Clustering
- Dimensionality reduction using PCA and UMAP.
- Clustering with K-Means and Spectral Clustering.
- Evaluation using:
  - Silhouette Score
  - Davies-Bouldin Score
  - Calinski-Harabasz Index

### 4. Recommendation System
- Leveraged cosine similarity on reduced feature embeddings.
- Implemented K-Nearest Neighbors (KNN) for personalized recommendations.

### 5. Visualization
- 2D and 3D embeddings using UMAP and t-SNE.
- Cluster visualizations with labeled data points.

## Key Results
- **Optimal Clustering**: Achieved using K-Means with 28 clusters.
- **Recommendation Accuracy**: Effective recommendations based on nearest neighbors within clusters.
- **Dimensionality Reduction**: UMAP provided meaningful low-dimensional embeddings for visualization and clustering.

## Requirements
- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `umap-learn`
- `spacy` (with `en_core_sci_lg`)
- `tensorflow`
