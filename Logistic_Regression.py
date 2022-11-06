# Import libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Logistic Regression from scratch
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def main():
    # Import test and training data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_labels = pd.read_csv('train_result.csv')

    # Create a new column in the train dataframe that contains the result of the train_result dataframe
    train['result'] = train_labels['Class']

    # Create a new dataframe that contains only the columns that we want to use
    df_features = train.iloc[:, :1568]
    df_label = train.iloc[:, 1569]
    df_test = test.iloc[:, :1568]

    # Train the model
    regressor = LogisticRegression(lr=0.0001, n_iters=1000)
    regressor.fit(df_features, df_label)

    # Predict the test set results
    y_pred = regressor.predict(df_test)

    # Predict accuracy for the test set
    print("Accuracy: ", regressor.score(df_test, y_pred))


if __name__ == "__main__":
    main()