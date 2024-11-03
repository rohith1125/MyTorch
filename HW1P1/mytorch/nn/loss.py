import numpy as np
from .activation import Softmax

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) ** 2  # TODO
        sse = np.ones(self.N).T @ se @ np.ones(self.C)  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]

        Ones_C = np.ones((C, 1), dtype='f')
        Ones_N = np.ones((N, 1), dtype='f')
        
        # # Softmax calculation
        # exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))  # For numerical stability
        # self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)  # Softmax probabilities

        # # Cross-entropy calculation
        # crossentropy = -Y * np.log(self.softmax + 1e-12)  # Add small constant for stability
        # sum_crossentropy = np.sum(crossentropy)  # Sum of cross-entropy terms
        # L = sum_crossentropy / N  # Mean cross-entropy loss
        
        # self.softmax = None #TODO
        # crossentropy = None #TODO
        # sum_crossentropy = None #TODO
        # L = None #TODO
        # Softmax probabilities for stability
        softmax = Softmax()
        self.softmax = softmax.forward(A)

        # Calculate cross-entropy
        crossentropy = -np.sum(Y * np.log(self.softmax), axis=1)  # Shape: (N,)
        # Sum over all samples
        sum_crossentropy = np.sum(crossentropy)
        # Mean cross-entropy loss
        L = sum_crossentropy / N
        return L

    def backward(self):
        dLdA = (self.softmax - self.Y) / self.A.shape[0]  # Divide by N for mean
        # dLdA = (self.softmax - self.Y) / self.N
        #dLdA = None #TODO

        return dLdA