import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
from GL_functions import GL_functions



class GL_calculator(GL_functions):
    '''
    This Class will utilise functions defined in GL_function class and have two type of method.
    1) Performing Integrals
    2) Ploting Graphs

    INtegrals are necessarly calculation for bayesian inference in marginalisation step.
    > If integral is over all the parameters we get probability distribution of model
    > If integral is over all parameters except one. The result will give probability distribution of left out variable.

    fill folling dictinary to instantiate class

    Priors = {'bins': , 'rmin': , 'rmax': , 'w_min': , 'w_max': , 'w_resolution': }


    Make sure data is numpy array with 3 columns 

    '''
    def __init__(self, data, priors):
        self.data = data
        self.priors = priors
        self.phi_limits = [0, 2 * np.pi]
        self.w_values = np.linspace(self.priors['w_min'], self.priors['w_max'], self.priors['w_resolution'])
        self.m = self.priors['bins']
        self.rmin = self.priors['rmin']
        self.rmax = self.priors['rmax']
        self.b_values = np.linspace(2, self.m, self.m - 1).astype(int)
        self.res = []
        self.prob_m = []
        self.prob_w = []
        #data and priors are dictionary specified when class is instantiated
        # m is max bins ,w_min/ w_max, r_min, r_max are specified in priors
        #W_values are discrete values of w where calculation is performed
        #b_values are diffrent model with number of bins where calculation is performed
        # Last three are class variable, which will be calculated when class methods are performed on data
       

    def Pw_dm(self):
        # Integral over phi, will give normalised probability of frequency
        # Performed over all w_values and b_values

        self.res = [
            [
                quad(
                    lambda phi: self.liklihood(w, phi, b)[0], 
                    self.phi_limits[0], self.phi_limits[1], 
                    epsabs=1.0e-1
                )[0]  for w in self.w_values    
            ] for b in self.b_values
        ]
        # notice w is not integrated




    def Pd_m(self):
        #Integral over all parameters values w, phi, Gives likelihood of model
        #Performed for all the GL models with diffrent number of bins
                
        self.prob_m = [
            (
                dblquad(
                    lambda w, phi : self.liklihood(w, phi, b)[1], 
                    self.phi_limits[0], self.phi_limits[1], 
                    lambda w: self.priors['w_min'],
                    lambda w: self.priors['w_max'],
                    epsabs=1.0e-2
                )[0]
            ) for b in self.b_values
        ]



    
    def plot_data(self):
        #Ploting the raw data, used for calculation
              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')
        plt.vlines(self.data[:, 0], 0, self.data[:, 1], linewidth=0.5)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.grid(True)
        
        plt.show()
           

    def plot_Pw_m(self,specified_bin):
        #Ploting probability distribution of frequency(Unormalised) for a specific model. Argument take number of bins of model.
        #Cal be executed right after integral pw_dm, make sure calculation is done before executing
              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.w_values, (self.res[specified_bin-2]), 'o-')
        plt.xlabel('w')
        plt.ylabel('Unnormalised Probability')
        plt.show()
        
        
    def plot_Pd_m(self):
        #Plot probability denstion of all GL periodic model

        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.b_values, self.prob_m, 'o-')
        plt.xlabel('Bins')
        plt.ylabel('Unnormalised Probability')
        plt.show()
           

    def plot_Pw(self):
        # plot probability density of frequency after averaging over all the models 

        pio = np.array([[self.prob_m[b-2] * x for x in res_row] for b, res_row in zip(self.b_values, self.res)])  
        #pio = np.array(hio)
        kio = sum(pio[i-2] for i in self.b_values)
        nio = [x/sum(self.prob_m) for x in kio]
        self.prob_w = nio

        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.w_values, self.prob_w, 'o-')    
        plt.xlabel('W Values')
        plt.ylabel('Unnormalised Probability')
        plt.show()
 
      

class GL_sample:
    '''
    Plot the sample of GL models which are part of priors
    Can be execute after gl_calculator has been instantiated with data and priors.

    '''
    def __init__(self, gl_calculator):
        self.gl_calculator = gl_calculator
        self.w = np.random.uniform(self.gl_calculator.priors['w_min'], self.gl_calculator.priors['w_max'])
        self.phi = np.random.uniform(0, 2*np.pi)
        t_min = np.min(self.gl_calculator.data[:, 0])
        t_max = np.max(self.gl_calculator.data[:, 0])
        self.t = np.linspace(t_min, t_max, 500)
        self._plot()

    def _j(self, t):
        return np.floor(1 + ((self.gl_calculator.m) * (np.mod((self.w)*t+self.phi,2*np.pi)) / (2*np.pi)))

    def _r(self, t, *r_values):
        if len(r_values) != self.gl_calculator.m:
            raise ValueError(f"Expected {self.gl_calculator.m} r_values, got {len(r_values)}")
        index = int(self._j(t)) - 1  # Subtract 1 to match Python's 0-indexing
        return r_values[index]

    def _plot(self):
        r_values = np.random.randint(self.gl_calculator.rmin, self.gl_calculator.rmax, size=self.gl_calculator.m)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.t, [self._r(time, *r_values) for time in self.t])
        plt.xlabel('time')
        plt.ylabel('flux')
        plt.title('Function Plot')
        plt.show()
