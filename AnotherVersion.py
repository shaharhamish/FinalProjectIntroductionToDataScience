import pandas as pd
import numpy as np
import math

# Load the dataset
df = pd.read_csv('breast-cancer.csv')  # Load the breast cancer dataset

# Remove the 'id' column (not a predictive feature)
df = df.drop(columns=['id'])

# Convert column names to lowercase and replace spaces with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Map the diagnosis column: 'B' -> 0 (Benign), 'M' -> 1 (Malignant)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Handle missing values by filling them with the mean of each column
df.fillna(df.mean(), inplace=True)

# Split data into features (X) and target (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Shuffle and split the data into train, validation, and test sets (60%, 20%, 20%)
indices = np.random.permutation(len(df))
train_size = int(0.6 * len(df))
val_size = int(0.2 * len(df))

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train, y_train = X.iloc[train_indices].copy(), y.iloc[train_indices].copy()
X_val, y_val = X.iloc[val_indices].copy(), y.iloc[val_indices].copy()
X_test, y_test = X.iloc[test_indices].copy(), y.iloc[test_indices].copy()

# Define the numeric columns for normalization and distance calculation
numeric_columns = X_train.columns.tolist()

# Apply Min-Max normalization to numeric columns
min_vals = {}
max_vals = {}

for col in numeric_columns:
    min_vals[col] = X_train[col].min()
    max_vals[col] = X_train[col].max()
    # Normalize training, validation, and test sets
    X_train.loc[:, col] = (X_train[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
    X_val.loc[:, col] = (X_val[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
    X_test.loc[:, col] = (X_test[col] - min_vals[col]) / (max_vals[col] - min_vals[col])


# Function to calculate Euclidean distance between two samples
def euclidean_distance(point1, point2, numeric_cols):
    squared_diff_sum = 0.0
    for col in numeric_cols:
        diff = point1[col] - point2[col]
        squared_diff_sum += diff ** 2
    return math.sqrt(squared_diff_sum)


# Function to find the nearest neighbor (1-NN) of a test sample
def find_1nn(test_sample, train_data, numeric_cols):
    min_distance = float('inf')
    nearest_label = None

    # Iterate through each training sample to find the closest
    for _, train_sample in train_data.iterrows():
        distance = euclidean_distance(test_sample, train_sample, numeric_cols)
        if distance < min_distance:
            min_distance = distance
            nearest_label = train_sample['diagnosis']

    return nearest_label, min_distance


# KNN classifier (using 1-NN logic here)
def knn(X_train, y_train, X_test, k):
    y_pred = []
    # Combine training features and labels for easy access
    train_data = X_train.copy()
    train_data['diagnosis'] = y_train

    # Predict the label for each test sample
    for _, test_sample in X_test.iterrows():
        label, _ = find_1nn(test_sample, train_data, numeric_columns)
        y_pred.append(label)

    return np.array(y_pred)


# Gaussian Naive Bayes classifier
def gaussian_naive_bayes(X_train, y_train, X_test):
    # Calculate mean and variance for each feature for each class
    means = X_train.groupby(y_train).mean()
    variances = X_train.groupby(y_train).var()
    class_priors = y_train.value_counts(normalize=True)

    def calculate_probability(x, mean, var):
        # Calculate the probability density function for Gaussian distribution
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(sample):
        class_probabilities = {}
        for class_label in means.index:
            class_prob = class_priors[class_label]
            for col in numeric_columns:
                class_prob *= calculate_probability(sample[col], means.loc[class_label, col], variances.loc[class_label, col])
            class_probabilities[class_label] = class_prob
        return max(class_probabilities, key=class_probabilities.get)

    # Predict X_test using the trained Gaussian Naive Bayes classifier
    predictions = X_test.apply(predict, axis=1)
    return predictions.values


# Function to calculate entropy
def entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Function to calculate information gain for a categorical attribute
def information_gain_categorical(X, y, attribute):
    unique_values = X[attribute].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset_y = y[X[attribute] == value]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    return entropy(y) - weighted_entropy


# Function to calculate information gain for a real-valued attribute
def information_gain_real(X, y, attribute):
    # Sort the data based on the attribute values
    sorted_indices = X[attribute].argsort()
    sorted_X, sorted_y = X.iloc[sorted_indices], y.iloc[sorted_indices]
    sorted_y = sorted_y.reset_index(drop=True)  # Reset index to avoid KeyError

    best_info_gain = -1
    best_threshold = None

    # Iterate through possible split points to find the best threshold
    for i in range(1, len(sorted_y)):
        if sorted_y[i] != sorted_y[i - 1]:
            # Calculate the threshold as the midpoint between two consecutive values
            threshold = (sorted_X[attribute].iloc[i] + sorted_X[attribute].iloc[i - 1]) / 2

            # Split the data into left and right subsets
            left_y = sorted_y[:i]
            right_y = sorted_y[i:]

            # Calculate the information gain
            info_gain = entropy(y) - (
                        (len(left_y) / len(y)) * entropy(left_y) + (len(right_y) / len(y)) * entropy(right_y))

            # Update the best information gain and threshold
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold

    return best_info_gain, best_threshold


# Recursive function to learn an unpruned decision tree
def LearnUnprunedTree(X, y):
    # If all records have identical values in all attributes or all values in y are the same, return a Leaf Node
    if len(np.unique(y)) == 1 or all(X.nunique() == 1):
        return {'type': 'leaf', 'class': y.mode()[0]}

    # Calculate information gain for each attribute
    best_info_gain = -1
    best_attribute = None
    best_threshold = None
    is_categorical = True

    for attribute in X.columns:
        if attribute in numeric_columns:
            info_gain, threshold = information_gain_real(X, y, attribute)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
                best_threshold = threshold
                is_categorical = False
        else:
            info_gain = information_gain_categorical(X, y, attribute)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
                best_threshold = None
                is_categorical = True

    # Split the dataset on the best attribute
    if is_categorical:
        tree = {'type': 'node', 'attribute': best_attribute, 'children': {}}
        for value in X[best_attribute].unique():
            subset_X = X[X[best_attribute] == value]
            subset_y = y[X[best_attribute] == value]
            tree['children'][value] = LearnUnprunedTree(subset_X, subset_y)
    else:
        tree = {'type': 'node', 'attribute': best_attribute, 'threshold': best_threshold, 'left': None, 'right': None}
        left_X = X[X[best_attribute] <= best_threshold]
        left_y = y[X[best_attribute] <= best_threshold]
        right_X = X[X[best_attribute] > best_threshold]
        right_y = y[X[best_attribute] > best_threshold]
        tree['left'] = LearnUnprunedTree(left_X, left_y)
        tree['right'] = LearnUnprunedTree(right_X, right_y)

    return tree


# Function to classify a sample using the decision tree
def classify(tree, sample):
    if tree['type'] == 'leaf':
        return tree['class']
    if 'threshold' in tree:
        if sample[tree['attribute']] <= tree['threshold']:
            return classify(tree['left'], sample)
        else:
            return classify(tree['right'], sample)
    else:
        value = sample[tree['attribute']]
        if value in tree['children']:
            return classify(tree['children'][value], sample)
        else:
            # If the value is not seen during training, return the mode of the training target
            return y_train.mode()[0]


# Decision Tree classifier
def decision_tree(X_train, y_train, X_test):
    tree = LearnUnprunedTree(X_train, y_train)
    y_pred = [classify(tree, sample) for _, sample in X_test.iterrows()]
    return np.array(y_pred)


# Function to evaluate and print metrics
def evaluate_model(y_true, y_pred, set_name):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) != 0 else 0
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) != 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(
        f"{set_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F-measure: {f_measure:.2f}")


# Main function to run the classifiers
def main():
    k = 5  # Number of neighbors for KNN

    # KNN Classifier
    y_pred_knn_val = knn(X_train, y_train, X_val, k)
    y_pred_knn_test = knn(X_train, y_train, X_test, k)
    evaluate_model(y_val, y_pred_knn_val, "KNN Validation")
    evaluate_model(y_test, y_pred_knn_test, "KNN Test")

    # Naive Bayes Classifier
    y_pred_nb_val = gaussian_naive_bayes(X_train, y_train, X_val)
    y_pred_nb_test = gaussian_naive_bayes(X_train, y_train, X_test)
    evaluate_model(y_val, y_pred_nb_val, "Naive Bayes Validation")
    evaluate_model(y_test, y_pred_nb_test, "Naive Bayes Test")

    # Decision Tree Classifier
    y_pred_dt_val = decision_tree(X_train, y_train, X_val)
    y_pred_dt_test = decision_tree(X_train, y_train, X_test)
    evaluate_model(y_val, y_pred_dt_val, "Decision Tree Validation")
    evaluate_model(y_test, y_pred_dt_test, "Decision Tree Test")


if __name__ == "__main__":
    main()