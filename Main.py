import pandas as pd
import numpy as np
import math

# Load the dataset
df = pd.read_csv('breast-cancer.csv')

# Remove the 'id' column
df = df.drop(columns=['id'])

# Convert column names to lowercase and replace spaces with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Map the diagnosis column: 'B' -> 0, 'M' -> 1
'''In this part I change the diagnosis column to 0 and 1 instead of B and M
B mean benign and M mean malignant'''
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Handle missing values by filling with mean
df.fillna(df.mean(), inplace=True)

# Split into features (X) and target (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Shuffle the indices
indices = np.random.permutation(len(df))
#TODO: split the data to train, test and val

def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
#TODO: KNN function
def knn(X_train, y_train, X_test, k):
    print("Training KNN classifier with k =", k)

#TODO: Naïve Bayes function
def naive_bayes(X_train, y_train, X_test):
    print("Training Naive Bayes classifier")

#TODO: Decision tree function
def decisionTree(X_train, y_train, X_test):
    print("Training Decision Tree classifier")