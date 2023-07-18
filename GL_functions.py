import numpy as np
from scipy.special import erfc




class GL_functions:#This is Model Used to generate Bins Structure in GL Method
    '''GL Method has two defining functions first is the model which makes periodic bins, and Second is the liklihood function.
    Model is Defined as j(t)= int[1+m(wt+phi)mod2pi/2pi]

     Liklyhood is split into several parts wj, dw, d2w and kai_square (Look paper for theier meaning)

     refer (gregory 1992)

     This code Assumes Data is numpy array with three column (Time, Flux, and Error)


    '''
    def j(self, t, w, phi, b):  # Model
        return (1 + ((b * (np.mod(w * t + phi, 1)) )/ (1))).astype(int)
        # It  create periodic m distinct bins repeting over period w/2pi 

    def liklihood(self, w, phi, b):  #Liklihood

        #first step is to batch the data J-values will take time component of data to batch them.
        j_values = self.j(self.data[:, 0], w, phi, b)


        #Full liklihood is product of  f1,f2 (binned dependent function) and c1,c2.. constant
        #f type functions call data in batches
        f1 = np.exp(
            -(1/2)*np.sum(
                [self.kai_square(j_values == p, w,phi) * self.wj(j_values == p, w, phi) for p in range(1, b +1)]
            )
        )  
        # m should be be provided in priors when class is instantiated  

        f2 = np.prod(
            [
                (self.wj(j_values == p, w, phi)**(-1/2))*
                (
                    erfc(self.yjmin(j_values == p, w, phi)) - erfc(self.yjmax(j_values == p, w, phi))
                )
              for p in range(1, b +1)
            ]
        )             
        


        C1 = ((2 * np.pi) ** (-len(self.data) / 2)) * ((np.pi / 2) ** (b / 2))
        #length of data is constant integer
        C2 = self.priors['rmax'] - self.priors['rmin'] ** (b)
        #rmin and rmax are float specified in priors when class is instantiated 
        C3 = np.prod(1 / self.data[:, 2])
        #product of reciprocal of error column of data is a constant
        C4 = 2 * np.pi * np.log(self.priors['w_max'] / self.priors['w_min'])
        C = (C1 * C3) / (C4*C2)
        #combining all the constants specified in GL-Paper

        return f1*f2*(1/w) , f1*f2*C*(1/w)
        # With/without constants, see use in gl_calculator class
    






    #J_index is argument used called the binned batch of data
    #for the code work flow read liklihood first, as Liklihood function calls daughter cells in batch of Bins
    #data(j_indics,1) means binned rows of data and second column ertries are called. 
    #data(j_indeces,1) is numpy array contaning all the flux lying in jth bin

    def wj(self, j_indices, w, phi): # sum of inverse of squared errors
        return np.sum(1 / (self.data[j_indices, 2] ** 2))

    def dw(self, j_indices, w, phi): # First Signature part of liklyhood calculation
        return np.sum(self.data[j_indices, 1] / (self.data[j_indices, 2] ** 2)) / self.wj(j_indices, w, phi)

    def d2w(self, j_indices, w, phi): # Second Signature part of liklihood calculation
        return np.sum((self.data[j_indices, 1] ** 2) / (self.data[j_indices, 2] ** 2)) / self.wj(j_indices, w, phi)

    def kai_square(self, j_indices, w, phi): # Kai_square part of liklihood calculation
        return self.d2w(j_indices, w, phi) - self.dw(j_indices, w, phi) ** 2

    def yjmin(self, j_indices, w, phi): # This Function is part of liklihood and require for Analytical Integral of r in marginalisation.
        return np.sqrt(self.wj(j_indices, w, phi) / 2) * (self.rmin - self.dw(j_indices, w, phi))
        #rmin will be specified in priors when class is instanciated

    def yjmax(self, j_indices, w, phi): # This Function is part of liklihood and require for Analytical Integral of r in marginalisation.
        return np.sqrt(self.wj(j_indices, w, phi) / 2) * (self.rmax - self.dw(j_indices, w, phi))
        #rmax will be specified in priors when class is instanciated


    