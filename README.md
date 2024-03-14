# Decision Tree and K-Nearest Neighbors (KNN) Implementation

This repository contains Python implementations of Decision Tree and K-Nearest Neighbors (KNN) algorithms from scratch. These algorithms are implemented without using any external libraries, providing a fundamental understanding of how they work.

## Decision Tree

A decision tree is a supervised learning algorithm used for classification and regression tasks. It works by recursively partitioning the feature space into regions, making decisions based on the feature values. This implementation includes the following features:

- **Binary Splitting**: The decision tree splits the feature space into two parts at each node.
- **Entropy and Information Gain**: It uses entropy and information gain to determine the best feature to split on at each node.
- **Pruning**: Implemented pruning techniques to avoid overfitting.
- **Classification and Regression**: Supports both classification and regression tasks.

### Usage

To use the Decision Tree implementation, follow these steps:

1. Import the `DecisionTree` class from `decision_tree.py`.
2. Create an instance of the `DecisionTree` class.
3. Fit the model to your training data using the `fit()` method.
4. Make predictions on new data using the `predict()` method.

Example:

```python
from decision_tree import DecisionTree

# Create an instance of Decision Tree
dt_classifier = DecisionTree()

# Fit the model to training data
dt_classifier.fit(X_train, y_train)

# Make predictions
predictions = dt_classifier.predict(X_test)

```

# Image Captioning using KNN with FAISS

This repository contains an implementation of image captioning using K-nearest neighbors (KNN) with FAISS, based on the paper "A Distributed Representation Based Query Expansion Approach for Image Captioning". The algorithm uses image embeddings and corresponding caption embeddings to predict captions for images.

## Introduction

Image captioning is a challenging task in computer vision and natural language processing. While Vision Language Models (VLMs) are popular for this task, this project explores the use of KNN with FAISS for image captioning, as suggested by earlier research.

### Algorithm Overview

1. **Input**: Image embeddings and corresponding caption embeddings (5 per image).
2. For each image:
   - Find the k nearest images using FAISS.
   - Compute the query vector as the weighted sum of the captions of the nearest images (k*5 captions per image).
3. Predict the caption by selecting the closest caption from the dataset to the query vector.

## Dataset

The MS COCO 2014 validation set is used for this project. The dataset contains images and corresponding captions.

## Implementation

The image and text embeddings are extracted from the CLIP model, although detailed knowledge about CLIP is not required for this implementation.

### Tasks Performed

1. Implement the algorithm and compute the BLEU score using FAISS for nearest neighbor computation. Starter code is provided in the repository.
2. Experiment with different values of k and record observations.
3. For a fixed k, try various options in the FAISS index factory to speed up computation in step 2. Record observations.
4. Qualitative Study: Visualize five images, their ground truth captions, and the predicted caption.

