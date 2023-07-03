import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
import time
import sqlite3
from GL_functions import GL_functions


class GL_sample:
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



    

class GL_calculator(GL_functions):
    def __init__(self, data, priors):
        self.data = data
        self.priors = priors
        self.phi_limits = [1, 2 * np.pi]
        self.w_values = np.linspace(self.priors['w_min'], self.priors['w_max'], self.priors['w_resolution'])
        self.m = self.priors['bins']
        self.rmin = self.priors['rmin']
        self.rmax = self.priors['rmax']
        self.m_values = np.linspace(2, self.m, self.m - 1).astype(int)
        self.res = []
        self.prob_data = []
        self.normalised_prob = []
        self.normalised_Pd_m = []
       

    def Pw_dm(self):
        
        self.res = [[quad(lambda phi: self.prob_w(w, phi), 
                          self.phi_limits[0], self.phi_limits[1], epsabs=1.0e-1)[0] 
                    for w in self.w_values] for m in self.m_values]

        
    def Pd_m(self):
        self.prob_data = []
        for mv in self.m_values:
            self.m = mv
            self.prob_data.append(dblquad(
                lambda w, phi: self.prob_w(w, phi),
                self.phi_limits[0], self.phi_limits[1],
                lambda phi: self.priors['w_min'],
                lambda phi: self.priors['w_max'],
                epsabs=1.0e-2
            )[0])
    
    def plot_data(self):
              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')
        plt.vlines(self.data[:, 0], 0, self.data[:, 1], linewidth=0.5)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.grid(True)
        
        plt.show()
           

    def plot_Pw_m(self,specified_bin):
              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.w_values, self.res[specified_bin], 'o-')
        plt.xlabel('w')
        plt.ylabel('Normalised Probability')
        plt.show()
        
        
    def plot_Pd_m(self):
        
        max_Pd_m = max(self.prob_data)
        
        self.normalised_Pd_m = [x / max_Pd_m for x in self.prob_data]
        
        fig = plt.figure(figsize=(12, 6))  # create a figure
        
        plt.plot(self.m_values, self.normalised_Pd_m, 'o-')
        
        plt.xlabel('Bins')
        plt.ylabel('Normalised Probability')
        
        plt.show()
           

    def plot_Pw(self):
        hio = [[self.prob_data[m-2] * x for x in res_row] for m, res_row in zip(self.m_values, self.res)]
        pio = np.array(hio)
        kio = sum(pio[i-2] for i in self.m_values)
        nio = [x/sum(self.prob_data) for x in kio]
        #Normalize Probability to max
        max_prob = max(nio) 
        self.normalised_prob = [x / max_prob for x in nio]

        fig = plt.figure(figsize=(12, 6))  # create a figure
        
        plt.plot(self.w_values, self.normalised_prob, 'o-')
        
        plt.xlabel('W Values')
        plt.ylabel('Normalised Probability')
        
        plt.show()
 
      