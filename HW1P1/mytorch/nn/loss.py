# Implementation by: Sai Rohith Tanuku
# Course Assignment - INFSCI 2595
import numpy as np
from .activation import Softmax  # Assuming Softmax is implemented in activation.py

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared Error (MSE) Loss.
        :param A: Output of the model, shape (N, C)
        :param Y: Ground-truth values, shape (N, C)
        :return: MSE Loss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # Number of samples
        self.C = A.shape[1]  # Number of classes

        # Compute squared error
        se = (A - Y) ** 2  # Element-wise squared error, shape (N, C)
        # Sum over all elements to get sum of squared errors
        sse = np.sum(se)  # Scalar
        # Calculate mean squared error
        mse = sse / (self.N * self.C)  # Scalar

        return mse

    def backward(self):
        """
        Calculate the gradient of the MSE Loss with respect to A.
        :return: Gradient dLdA, shape (N, C)
        """
        # Gradient of MSE loss with respect to A
        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)  # Shape (N, C)

        return dLdA


class CrossEntropyLoss:

    def __init__(self):
        # Initialize Softmax once for the class
        self.softmax = Softmax()

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss.
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values (one-hot encoded), shape (N, C)
        :return: Cross Entropy Loss (scalar)
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]  # Number of samples
        C = A.shape[1]  # Number of classes

        # Calculate softmax probabilities
        self.softmax_output = self.softmax.forward(A)

        # Calculate cross-entropy with numerical stability
        crossentropy = -np.sum(Y * np.log(self.softmax_output + 1e-12), axis=1)  # Shape: (N,)
        
        # Sum over all samples to get total cross-entropy, then average
        sum_crossentropy = np.sum(crossentropy)  # Sum of cross-entropy over samples
        L = sum_crossentropy / N  # Mean cross-entropy loss

        return L

    def backward(self):
        """
        Compute the gradient of the Cross Entropy Loss with respect to A.
        :return: Gradient dLdA, shape (N, C)
        """
        # Gradient of the cross-entropy loss with respect to input A
        dLdA = (self.softmax_output - self.Y) / self.A.shape[0]  # Shape (N, C)
        return dLdA
