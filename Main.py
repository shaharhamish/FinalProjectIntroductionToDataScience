import numpy as np
import pandas as pd

class ClassificationModels:
    """
    A class for implementing two machine learning classifiers:

    1. K-Nearest Neighbors (KNN):
       - Fits and predicts based on the nearest neighbors of a given data point.
    2. Naive Bayes:
       - Implements a probabilistic classifier based on Bayes' theorem, assuming independence between features.

    Methods:
        - fit_knn: Stores training data and labels for KNN.
        - predict_knn: Predicts labels for test data using KNN.
        - fit_naive_bayes: Trains Naive Bayes by calculating class probabilities and feature statistics.
        - predict_naive_bayes: Predicts labels for test data using Naive Bayes.
        - evaluate: Computes performance metrics (accuracy, precision, recall, and F-measure).
    """
    def __init__(self, k=3):
        # Initialize the classifier with k (number of neighbors for KNN)
        self.k = k
        self.data = None
        self.labels = None
        self.class_probabilities = {}  # For Naive Bayes
        self.feature_statistics = {}  # For Naive Bayes

    def fit_knn(self, data, labels):
        """
        Sets up the training data for the k-Nearest Neighbors (k-NN) model.

        Parameters:
        - data (numpy.ndarray): The training feature data.
        - labels (numpy.ndarray): The training labels corresponding to the feature data.
        """
        self.data = data
        self.labels = labels

    def predict_knn(self, test_data):
        # Predict labels for test data using the KNN algorithm
        predictions = []
        for test_point in test_data:
            # Calculate distances from the test point to all training points
            # The Euclidean distance is used: sqrt(sum((x_i - y_i)^2))
            distances = np.sqrt(np.sum((self.data - test_point) ** 2, axis=1))
            # Find the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            # Retrieve the labels of the nearest neighbors
            nearest_labels = self.labels[nearest_indices]
            # Predict the most common label among the neighbors
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)

    def fit_naive_bayes(self, data, labels):
        """
        Trains the Naive Bayes model by calculating class probabilities and feature statistics.

        This method computes the prior probabilities of each class and calculates the
        mean and standard deviation for each feature within each class.

        Parameters:
        - data (numpy.ndarray): The training feature data.
        - labels (numpy.ndarray): The training labels corresponding to the feature data.
        """
        self.data = data
        self.labels = labels
        classes = np.unique(labels)

        for cls in classes:
            # Extract data points belonging to the current class
            cls_data = data[labels == cls]
            # Calculate the probability of the class
            self.class_probabilities[cls] = len(cls_data) / len(data)
            # Calculate mean and standard deviation for each feature
            self.feature_statistics[cls] = {
                'mean': np.mean(cls_data, axis=0),
                'std': np.std(cls_data, axis=0) + 1e-9  # Add a small value to avoid division by zero
            }

    def predict_naive_bayes(self, test_data):
        # Predict labels for test data using the Naive Bayes algorithm
        predictions = []
        for test_point in test_data:
            class_probs = {}
            for cls in self.class_probabilities:
                # Retrieve the mean and standard deviation for the current class
                mean = self.feature_statistics[cls]['mean']
                std = self.feature_statistics[cls]['std']
                # Calculate the log-likelihood of the test point belonging to the class
                likelihood = -0.5 * np.sum(((test_point - mean) / std) ** 2 + np.log(2 * np.pi * std ** 2))
                class_probs[cls] = np.log(self.class_probabilities[cls]) + likelihood
            # Predict the class with the highest probability
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

    def evaluate(self, predictions, true_labels):
        # Evaluate the performance of the classifier using accuracy, precision, recall, and F-measure
        tp = np.sum((predictions == 1) & (true_labels == 1))  # True positives
        tn = np.sum((predictions == 0) & (true_labels == 0))  # True negatives
        fp = np.sum((predictions == 1) & (true_labels == 0))  # False positives
        fn = np.sum((predictions == 0) & (true_labels == 1))  # False negatives

        accuracy = (tp + tn) / len(true_labels)  # Overall accuracy
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F-measure

        return accuracy, precision, recall, f_measure

def main():
    # Load dataset (replace with actual dataset from Kaggle)
    data = pd.read_csv('breast-cancer.csv')  # Replace 'breast_cancer.csv' with the actual file

    # Ensure labels are numerical (e.g., 'M' -> 1, 'B' -> 0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Extract features and labels
    features = data.iloc[:, 2:].values  # Assuming the first two columns are ID and Diagnosis
    labels = data['diagnosis'].values

    # Normalize features to bring them to a similar scale
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

    # Split dataset into train and test
    split_ratio = 0.8  # Percentage of data to use for training
    split_index = int(len(features) * split_ratio)
    train_data, test_data = features[:split_index], features[split_index:]  # Split features
    train_labels, test_labels = labels[:split_index], labels[split_index:]  # Split labels

    model = ClassificationModels(k=3)  # Initialize the model with k=3 for KNN

    # KNN
    model.fit_knn(train_data, train_labels)  # Train KNN
    knn_predictions = model.predict_knn(test_data)  # Predict using KNN
    knn_metrics = model.evaluate(knn_predictions, test_labels)  # Evaluate KNN performance
    print("KNN Metrics (Accuracy, Precision, Recall, F-Measure):", knn_metrics)

    # Naive Bayes
    model.fit_naive_bayes(train_data, train_labels)  # Train Naive Bayes
    nb_predictions = model.predict_naive_bayes(test_data)  # Predict using Naive Bayes
    nb_metrics = model.evaluate(nb_predictions, test_labels)  # Evaluate Naive Bayes performance
    print("Naive Bayes Metrics (Accuracy, Precision, Recall, F-Measure):", nb_metrics)

if __name__ == "__main__":
    main()
