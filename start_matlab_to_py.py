import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Load the pixel data
data = np.loadtxt('mfeat-pix.txt')

# Function to plot digit images
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            pic = data[200 * i + j, :].reshape(16, 15).T  # Reshape to 16x15 and transpose
            axes[i, j].imshow(pic, cmap='gray', vmin=0, vmax=6)  
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# Plot the digits
plot_digits(data)

# Split the data into a training and a testing dataset
train_indices = np.concatenate([np.arange(100) + 200 * i for i in range(10)])
test_indices = train_indices + 100
train_patterns = data[train_indices, :]
test_patterns = data[test_indices, :]

# Create indicator matrices size 10 x 1000 with the class labels coded by binary indicator vectors
train_labels = np.eye(10).repeat(100, axis=0).T
test_labels = train_labels.copy()

# Create a row vector of correct class labels (from 1 ... 10 for the 10 classes). This vector is the same for training and testing.
correct_labels = np.concatenate([np.full(100, i + 1) for i in range(10)])

# Compute the mean images for each class
mean_train_images = np.array([train_patterns[100 * i: 100 * (i + 1), :].mean(axis=0) for i in range(10)]).T

# Compute feature values for training and testing datasets
feature_values_train = mean_train_images.T @ train_patterns.T
feature_values_test = mean_train_images.T @ test_patterns.T

# Compute linear regression weights W
W = np.linalg.inv(feature_values_train @ feature_values_train.T) @ feature_values_train @ train_labels.T

# Compute train misclassification rate
classification_hypotheses_train = W.T @ feature_values_train
predicted_train_labels = np.argmax(classification_hypotheses_train, axis=0) + 1
train_accuracy = accuracy_score(correct_labels, predicted_train_labels)
print(f'Train misclassification rate = {1 - train_accuracy:.3f}')

# Compute test misclassification rate
classification_hypotheses_test = W.T @ feature_values_test
predicted_test_labels = np.argmax(classification_hypotheses_test, axis=0) + 1
test_accuracy = accuracy_score(correct_labels, predicted_test_labels)
print(f'Test misclassification rate = {1 - test_accuracy:.3f}')

# Ridge regression on raw images
alpha = 10000
ridge = Ridge(alpha=alpha, fit_intercept=False)
ridge.fit(train_patterns, train_labels.T)

# Compute train misclassification rate for ridge regression
predicted_train_labels_ridge = np.argmax(ridge.predict(train_patterns), axis=1) + 1
train_accuracy_ridge = accuracy_score(correct_labels, predicted_train_labels_ridge)
print(f'Train misclassification rate ridge = {1 - train_accuracy_ridge:.3f}')

# Compute test misclassification rate for ridge regression
predicted_test_labels_ridge = np.argmax(ridge.predict(test_patterns), axis=1) + 1
test_accuracy_ridge = accuracy_score(correct_labels, predicted_test_labels_ridge)
print(f'Test misclassification rate ridge = {1 - test_accuracy_ridge:.3f}')