import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
import time
import sqlite3




class GL_functions:
    def j(self, t, w, phi):  # Function to specify bin
        return np.floor(1 + (self.m * (np.mod(w * t + phi, 2 * np.pi)) / (2 * np.pi))).astype(int)
    
    def wj(self, j_indices, w, phi): # Inverse sum of errors
        return np.sum(1 / (self.data[j_indices, 2] ** 2))

    def dw(self, j_indices, w, phi): # First Signature
        return np.sum(self.data[j_indices, 1] / (self.data[j_indices, 2] ** 2)) / self.wj(j_indices, w, phi)

    def d2w(self, j_indices, w, phi): # Second Signature
        return np.sum((self.data[j_indices, 1] ** 2) / (self.data[j_indices, 2] ** 2)) / self.wj(j_indices, w, phi)

    def kai_square(self, j_indices, w, phi): # Kai_square
        return self.d2w(j_indices, w, phi) - self.dw(j_indices, w, phi) ** 2

    def yjmin(self, j_indices, w, phi): # Define Arbitrary yjmin
        return np.sqrt(self.wj(j_indices, w, phi) / 2) * (self.rmin - self.dw(j_indices, w, phi))

    def yjmax(self, j_indices, w, phi): # Define Arbitrary yjmax
        return np.sqrt(self.wj(j_indices, w, phi) / 2) * (self.rmax - self.dw(j_indices, w, phi))


    def prob_w(self, w, phi):  # Define probability of w
        j_values = self.j(self.data[:, 0], w, phi)

        f1 = np.exp(-(1 / 2) * np.sum([self.kai_square(j_values == p, w, phi) *
                                        self.wj(j_values == p, w, phi) for p in range(1, self.m + 1)]))

        f2 = np.prod([np.sqrt(self.wj(j_values == p, w, phi)) *
                      (erfc(self.yjmin(j_values == p, w, phi)) - erfc(self.yjmax(j_values == p, w, phi)))
                      for p in range(1, self.m + 1)])

        C1 = ((2 * np.pi) ** (-len(self.data) / 2)) * ((np.pi / 2) ** (self.m / 2))
        C2 = (self.priors['rmax'] - self.priors['rmin']) ** self.m
        C3 = np.prod(1 / self.data[:, 2])
        C4 = 2 * np.pi * np.log(self.priors['w_max'] / self.priors['w_min'])
        C = (C1 * C2 * C3) / C4

        return f1 * f2 * (1 / w) * C
