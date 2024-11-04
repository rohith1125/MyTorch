# Implementation by: Sai Rohith Tanuku
# Course Assignment - INFSCI 2595
import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # Shape: (C_out x C_in)
        self.b = np.zeros((out_features, 1))            # Shape: (C_out x 1)
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A                                    # Shape: (N x C_in)
        self.N = A.shape[0]                          # store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = A @ self.W.T + self.Ones @ self.b.T      # Shape: (N x C_out)
        return Z

    def backward(self, dLdZ):
        """
        Backward pass on linear layer.
        
        :param dLdZ: Gradient of the loss w.r.t. Z, shape (N x C_out)
        :return: Gradient of the loss w.r.t. A (dLdA), shape (N x C_in)
        """
        # Calculate gradients w.r.t. A, W, and b
        dLdA = dLdZ @ self.W                    # Shape: (N x C_in)
        dLdW = dLdZ.T @ self.A                  # Shape: (C_out x C_in)
        dLdb = dLdZ.sum(axis=0).reshape(-1, 1)  # Shape: (C_out x 1)

        self.dLdW = dLdW
        self.dLdb = dLdb

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA