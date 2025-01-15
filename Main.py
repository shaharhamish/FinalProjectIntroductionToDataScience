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

# Improved Naive Bayes classifier
def naive_bayes(X_train, y_train, X_test):
    # Step 1: Convert training data to document format
    training_data = [(y_train.iloc[i], ' '.join(X_train.iloc[i].astype(str).values)) for i in range(len(y_train))]

    # Step 2: Calculate the total number of documents
    ptot = len(training_data)

    # Step 3: Count how many documents every class has
    pk = {}
    for (class_label, _) in training_data:
        if class_label not in pk:
            pk[class_label] = 1
        else:
            pk[class_label] += 1

    # Step 4: Create a dictionary with all unique words from all documents
    word_dict = set()
    for _, doc in training_data:
        words = doc.split()
        for word in words:
            word_dict.add(word.lower())  # Convert words to lowercase for consistency

    # Initialize dictionary to count how many documents each word appears in for every class
    pki = {word: {class_label: 0 for class_label in pk} for word in word_dict}

    # Step 5: Count how many documents each word appears in for every class
    for (class_label, doc) in training_data:
        words = set(doc.lower().split())  # Convert to lowercase and set for uniqueness
        for word in words:
            pki[word][class_label] += 1

    def predict(document):
        # Split the document into words, convert to lowercase
        words = set(document.lower().split())

        # Initialize a dictionary to store the log probability sum of each class
        class_log_probabilities = {}

        # Calculate the log probability of the document for each class
        for class_label in pk:
            # Start with the log probability of the class itself
            class_log_probabilities[class_label] = math.log(pk[class_label] / ptot)

            # Add the log probability of each word given the class
            for word in words:
                if word in word_dict:
                    # Use Laplace smoothing
                    word_prob = (pki[word][class_label] + 1) / (pk[class_label] + len(word_dict))
                    class_log_probabilities[class_label] += math.log(word_prob)

        # Predict the class with the highest log probability sum
        predicted_class = max(class_log_probabilities, key=class_log_probabilities.get)

        return predicted_class, class_log_probabilities

    # Predict X_test using the trained Naive Bayes classifier
    predictions = []
    for doc in X_test.astype(str).apply(lambda row: ' '.join(row.values), axis=1):
        predicted_class, _ = predict(doc)
        predictions.append(predicted_class)

    return np.array(predictions)

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
    sorted_indices = X[attribute].argsort()
    sorted_X, sorted_y = X.iloc[sorted_indices], y.iloc[sorted_indices]
    sorted_y = sorted_y.reset_index(drop=True)  # Reset index to avoid KeyError
    best_info_gain = -1
    best_threshold = None
    for i in range(1, len(sorted_y)):
        if sorted_y[i] != sorted_y[i - 1]:
            threshold = (sorted_X[attribute].iloc[i] + sorted_X[attribute].iloc[i - 1]) / 2
            left_y = sorted_y[:i]
            right_y = sorted_y[i:]
            info_gain = entropy(y) - ((len(left_y) / len(y)) * entropy(left_y) + (len(right_y) / len(y)) * entropy(right_y))
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

# Main function to run the classifiers
def main():
    k = 5  # Number of neighbors for KNN
    # KNN Classifier
    y_pred_knn_val = knn(X_train, y_train, X_val, k)
    accuracy_knn_val = np.mean(y_pred_knn_val == y_val)
    print(f"KNN Validation Accuracy: {accuracy_knn_val:.2f}")

    # Naive Bayes Classifier
    y_pred_nb_val = naive_bayes(X_train, y_train, X_val)
    accuracy_nb_val = np.mean(y_pred_nb_val == y_val)
    print(f"Naive Bayes Validation Accuracy: {accuracy_nb_val:.2f}")

    # Decision Tree Classifier
    y_pred_dt_val = decision_tree(X_train, y_train, X_val)
    accuracy_dt_val = np.mean(y_pred_dt_val == y_val)
    print(f"Decision Tree Validation Accuracy: {accuracy_dt_val:.2f}")

    # Evaluate all models on the test set
    y_pred_knn_test = knn(X_train, y_train, X_test, k)
    accuracy_knn_test = np.mean(y_pred_knn_test == y_test)
    print(f"KNN Test Accuracy: {accuracy_knn_test:.2f}")

    y_pred_nb_test = naive_bayes(X_train, y_train, X_test)
    accuracy_nb_test = np.mean(y_pred_nb_test == y_test)
    print(f"Naive Bayes Test Accuracy: {accuracy_nb_test:.2f}")

    y_pred_dt_test = decision_tree(X_train, y_train, X_test)
    accuracy_dt_test = np.mean(y_pred_dt_test == y_test)
    print(f"Decision Tree Test Accuracy: {accuracy_dt_test:.2f}")

if __name__ == "__main__":
    main()