import numpy as np


class SGD:

    def __init__(self, model, lr=0.1, momentum=0):

        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):

        # for i in range(self.L):

        #     if self.mu == 0:

        #         self.l[i].W = None  # TODO
        #         self.l[i].b = None  # TODO

        #     else:

        #         self.v_W[i] = None  # TODO
        #         self.v_b[i] = None  # TODO
        #         self.l[i].W = None  # TODO
        #         self.l[i].b = None  # TODO
        # for i in range(self.L):
        #     if self.mu == 0:
        #         # Standard SGD update without momentum
        #         self.l[i].W -= self.lr * self.l[i].dLdW  # Update weights
        #         self.l[i].b -= self.lr * self.l[i].dLdb  # Update biases
        #     else:
        #         # SGD update with momentum
        #         self.v_W[i] = self.mu * self.v_W[i] + self.lr * self.l[i].dLdW  # Update velocity for weights
        #         self.v_b[i] = self.mu * self.v_b[i] + self.lr * self.l[i].dLdb  # Update velocity for biases
        #         self.l[i].W -= self.v_W[i]  # Update weights with velocity
        #         self.l[i].b -= self.v_b[i]  # Update biases with velocity
        for i in range(self.L):
            if self.mu == 0:
                # Standard SGD update without momentum
                self.l[i].W -= self.lr * self.l[i].dLdW  # Update weights
                self.l[i].b -= self.lr * self.l[i].dLdb  # Update biases
            else:
                # SGD update with momentum
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW  # Update velocity for weights
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb  # Update velocity for biases
                self.l[i].W -= self.lr * self.v_W[i]  # Update weights with velocity
                self.l[i].b -= self.lr * self.v_b[i]  # Update biases with velocity