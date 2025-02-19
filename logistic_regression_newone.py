import numpy as np
import pandas as pd

class LogisticRegression:
    
    def transform( x, mean, standard_dev):
        """Apply feature scaling (normalization)."""
        return (x - mean) / standard_dev
    





    
    def normalization(self, X):
        """Compute mean and standard deviation for feature normalization."""
        m = len(X)
        mean = np.mean(X, axis=0)
        standard_dev = np.std(X, axis=0)
        
        new_array =self.transform(X, mean, standard_dev)
        return new_array, mean, standard_dev
    





    def oneencoding( self,y, numberofclasses):
        """One-hot encode the labels."""
        m = len(y)
        y_1 = np.zeros((m, numberofclasses))
        for i in range(m):
            y_1[i][y[i]] = 1
        return y_1
    






    def sigmoid(self,x, w, b):
        """Compute the sigmoid function."""
        z = np.dot(w, x) + b
        return 1.0 / (1.0 + np.exp(-z))
    






    def costfunction(self,x, y, w, b):
        """Compute the cost function for logistic regression."""
        m = x.shape[1]
        fwb = self.sigmoid(x, w, b)
        J = -np.sum(y * np.log(fwb) + (1 - y) * np.log(1 - fwb)) / m
        return J
    







    def gradient(self, x, y, w, b):
        """Compute the gradients for optimization."""
        m = x.shape[1]
        fwb = self.sigmoid(x, w, b)
        error = fwb - y

        dj_dw = np.dot(error, x.T) / m
        dj_db = np.sum(error, axis=1, keepdims=True) / m

        return dj_dw, dj_db
    








    def logisticregression(self, data_arr, alpha, iterations, numberclass):
        """Train the logistic regression model."""
        x = data_arr[:, 1:]  # Features
        y = data_arr[:, 0].astype(int)  # Labels

        # One-hot encode labels and transpose
        y_1 = self.oneencoding(y, numberclass).T

        # Normalize features
        x, mean, standard_dev = self.normalization(x)

        x = x.T  # Shape: (features, samples)
        w = np.zeros((y_1.shape[0], x.shape[0]))  # Shape: (classes, features)
        b = np.zeros((y_1.shape[0], 1))  # Bias for each class

        J_hist, w_hist, b_hist = [], [], []

        for i in range(iterations):
            dj_dw, dj_db = self.gradient(x, y_1, w, b)
            w -= alpha * dj_dw
            b -= alpha * dj_db
            J = self.costfunction(x, y_1, w, b)

            if i % 10 == 0:
                print(f"{i}th iteration completed")

            J_hist.append(J)
            w_hist.append(w.copy())
            b_hist.append(b.copy())

        return w, b, J_hist, mean, standard_dev







    def prediction(self,testdata, w, b, mean, standard_dev):
        """Predict the class label for new data."""
        x = testdata  # Test data should already be without labels
        x = self.transform(x, mean, standard_dev) 
        x = x.T  # Shape: (features, samples)

        y_pred = self.sigmoid(x, w, b)
        y_pred = (y_pred >= 0.5).astype(int)

        return y_pred
