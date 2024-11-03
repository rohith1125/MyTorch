import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))    # Sigmoid function, Shape: (N x C_out)
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)    # Derivative of sigmoid function, Shape: (N x C_out)
        dLdZ = dLdA * dAdZ              # Derivative of loss w.r.t. sigmoid output, Shape: (N x C_out) 
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = np.tanh(Z)              # Tanh function, Shape: (N x C_out)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2            # Derivative of tanh function, Shape: (N x C_out)
        dLdZ = dLdA * dAdZ                # Derivative of loss w.r.t. tanh output, Shape: (N x C_out)
        return dLdZ

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.Z = Z                            # Input, Shape: (N x C_in)
        self.A = np.maximum(0, self.Z)          # ReLU function, Shape: (N x C_out)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0)    # Derivative of ReLU function, Shape: (N x C_out)
        dLdZ = dLdA * dAdZ                  # Derivative of loss w.r.t. ReLU output, Shape: (N x C_out)
        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
        self.Z = Z                            # Input, Shape: (N x C_in)
        self.A = 0.5 * self.Z * (1 + scipy.special.erf(Z / np.sqrt(2))) # GELU function, Shape: (N x C_out)
        return self.A
    
    def backward(self, dLdA):
        dAdZ = 0.5 * (1 + scipy.special.erf(self.Z / np.sqrt(2))) + (self.Z / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z**2) # Derivative of GELU function, Shape: (N x C_out)
        dLdZ = dLdA * dAdZ                                                                                                # Derivative of loss w.r.t. GELU output, Shape: (N x C_out)
        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        # Subtract max(Z) along each row for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))    # Shape: (N x C_out)
        # Sum of exponents for each row
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)        # Shape: (N x 1)
        # Softmax probabilities
        self.A = exp_Z / sum_exp_Z                             # Shape: (N x C_out)
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = self.A.shape[0]
        C = self.A.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros(self.A.shape, dtype="f")                # Shape: (N x C_out)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C), dtype="f")                    # Shape: (C_out x C_out)

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    J[m,n] = self.A[i,m] * (1 - self.A[i,n]) if m == n else -self.A[i,m] * self.A[i,n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.dot(J, dLdA[i])                      # Shape: (C_out)

        return dLdZ
'''
import numpy as np
from scipy.special import erf, expit  # expit is the sigmoid function

class Identity:
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    def forward(self, Z):
        # Sigmoid activation function: 1 / (1 + exp(-Z))
        self.A = expit(Z)  # More stable and numerically efficient
        return self.A

    def backward(self, dLdA):
        # Sigmoid derivative: A * (1 - A)
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    def forward(self, Z):
        # Tanh activation function: (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        # Tanh derivative: 1 - A^2
        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    def forward(self, Z):
        # ReLU activation function: max(0, Z)
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        # ReLU derivative: 1 where A > 0, else 0
        dAdZ = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:
    def forward(self, Z):
        # GELU activation function: 0.5 * Z * (1 + erf(Z / sqrt(2)))
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        # GELU derivative based on provided formula:
        dAdZ = 0.5 * (1 + erf(Z / np.sqrt(2))) + (Z / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax:
    def forward(self, Z):
        """
        Softmax function for stability:
        softmax(Z) = exp(Z - max(Z)) / sum(exp(Z - max(Z)))
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        Softmax backward pass:
        dLdZ = dLdA * Jacobian of softmax, calculated for each row.
        """
        N, C = dLdA.shape
        dLdZ = np.zeros_like(dLdA)

        for i in range(N):
            # Calculate Jacobian for each row
            a = self.A[i].reshape(-1, 1)  # Shape: (C, 1)
            J = np.diagflat(a) - np.dot(a, a.T)  # J = diag(a) - a * a^T

            # Calculate dLdZ for each row
            dLdZ[i, :] = np.dot(J, dLdA[i])

        return dLdZ

'''