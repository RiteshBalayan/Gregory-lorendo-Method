import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared

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
        C2 = (self.rmax - self.rmin) ** (b)
        #rmin and rmax are float specified in priors when class is instantiated 
        C3 = np.prod(1 / self.data[:, 2])
        #product of reciprocal of error column of data is a constant
        C4 = 2 * np.pi * np.log(self.w_max / self.w_min)
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

class Oscilation_Detector(GL_functions):
    '''

    Make sure data is numpy array with 3 columns 

    '''
    def __init__(self, data, w_min, w_max):
        self.data = data
        self.phi_limits = [0, 2 * np.pi]
        self.w_min = w_min
        self.w_max = w_max
        self.GL_resolution = 500 # Default Value
        self.GP_resolution = 200 # Default Value
        self.w_values = np.linspace(self.w_min, self.w_max, self.GL_resolution)
        self.w_values_GL = np.linspace(self.w_min, self.w_max, self.GL_resolution) #To avoid w_values change in other methods affect results
        self.m = 7 # Default Value
        self.m_GL = 7
        self.m_GP =7
        self.rmax = max(self.data[:,1]) + abs(max(self.data[:,1])/5)
        self.rmin = min(self.data[:,1]) - abs(max(self.data[:,1])/5)
        self.b_values = np.linspace(2, self.m, self.m - 1).astype(int)
        self.time = np.linspace(0, self.data[-1,0], 1000)#high resolution time for ploting only
        self.freq = [] # FFT
        self.power = [] #FFT
        self.log_prob_m_matrix = []
        self.power_mask = []
        self.GL = []
        self.GP_period = []
        self.GP_liklehood = []
        self.fft_result = 0
        self.GL_result = 0
        self.GP_result = 0
        #data and priors are dictionary specified when class is instantiated
        # m is max bins ,w_min/ w_max, r_min, r_max are specified in priors
        #W_values are discrete values of w where calculation is performed
        #b_values are diffrent model with number of bins where calculation is performed
        # Last three are class variable, which will be calculated when class methods are performed on data

    def plot_data(self):
        #Ploting the raw data, used for calculation              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')
        plt.vlines(self.data[:, 0], 0, self.data[:, 1], linewidth=0.5)
        plt.xlabel('Time', color='black')
        plt.ylabel('Flux', color='black')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.show()

    def Compute_FFT(self):

        # Find defining Parameters of FFT from Data
        dt = float(self.data[1:2,0]) - float(self.data[0:1,0])
        N = len(self.data[:,0])
        #Compute FFT
        F = np.fft.fft(self.data[:,1])
        F = np.abs(F[:N//2])  # take only the positive-frequency terms
        freq = np.fft.fftfreq(N, dt)[:N//2]  # compute the frequencies  
        #Compute power spectral density
        power = np.abs(F)**2
        # Mask on our required range
        mask = (freq >= self.w_min) & (freq <= self.w_max)
        #Save Results and raw Data
        self.power_mask = power[mask]
        self.freq = freq
        self.power = power
        self.fft_result = self.freq[mask][np.nanargmax(self.power[mask])]

    def Plot_FFT(self):

        #Change FFT Results Shape
        freq = np.array(self.freq)
        power = np.array(self.power)
        #plottting
        plt.figure(figsize=(12,6))
        plt.title('')
        plt.xlabel('Frequency [Hz]', fontsize=14, color='black')
        plt.ylabel('Power', fontsize=14, color='black')
        #only plot the frequencies in range(w_min, w_max)
        # Mask on our required range
        mask = (freq >= self.w_min) & (freq <= self.w_max)
        plt.plot(freq[mask], power[mask], color='black')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()    

    def Compute_GL(self, bins = None,  resolution = None):

        # if bins and resolution are not provided, use the default variables
        if bins is not None:
            self.b_values = np.linspace(2, bins, bins - 1).astype(int)
            self.m_GL = bins

        if resolution is not None:
            self.w_values_GL = np.linspace(self.w_min, self.w_max, resolution)

        #First Compute log likelihood for all the discrete values of w and bins
        self.log_prob_m_matrix = [[np.log(self.liklihood(w, 1.3, b)[0]) for w in self.w_values_GL]for b in self.b_values]
        #Take Linear Superposition of diffrent Bins
        self.GL = np.sum(np.array(self.log_prob_m_matrix), axis=0)
        #Save best frequency of GL
        self.GL_result = self.w_values_GL[np.nanargmax(self.GL)]  #nanargmax will work because argmax can give nan values

    def Plot_GL(self):

        fig = plt.figure(figsize=(12, 6))
        # Plotting the data w with respect to superimposed log liklihood of diffrent models
        plt.plot(self.w_values_GL, self.GL, color='black')
        # Adding labels and title
        plt.title('', fontsize=16, color='black')
        plt.xlabel('Frequency [Hz]', fontsize=14, color='black')
        plt.ylabel('Log Likelihood', fontsize=14, color='black')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()  


    def Compute_GP(self, resolution = None):

            #Use default Resolution if non Provided
            if resolution is None:
                resolution = self.GP_resolution

            #reshape data for Gausian process 
            X_Train = self.data.T[0].reshape(-1,1)
            Y_Train = self.data.T[1].reshape(-1,1)
            # Initialize lists to store the periods and corresponding log-marginal likelihoods
            fine_periods = []
            fine_log_marginal_likelihoods = []
            #find period in terms of frequency
            min_period = 1/self.w_max
            max_period = 1/self.w_min
            w_values_GP = np.arange(1/self.w_max, 1/self.w_min, (max_period-min_period)/resolution)
            # Loop over periods
            for period in w_values_GP:
                # Define the kernel
                kernel = ExpSineSquared(length_scale=1.0, periodicity=period,
                                        periodicity_bounds=(period, period+0.001))  + WhiteKernel(noise_level=self.data.T[2][0])
                # Fit the Gaussian process regressor
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0).fit(X_Train, Y_Train)
                # Compute the log-marginal likelihood of the optimized fit
                log_marginal_likelihood = gp.log_marginal_likelihood()
                # Store the period and corresponding log-marginal likelihood
                fine_periods.append(period)
                fine_log_marginal_likelihoods.append(log_marginal_likelihood)
            #save best period as GP Result
            self.GP_result = 1/fine_periods[np.nanargmax(np.array(fine_log_marginal_likelihoods))]
            #Save raw data to be used else where in class
            self.GP_period = fine_periods
            self.GP_liklehood = fine_log_marginal_likelihoods

    def Plot_GP(self):
        
        # Plot the log-marginal likelihood as a function of the period
        fig = plt.figure(figsize=(12, 6))  # create a figure
        fine_frequency = 1/np.array(self.GP_period)
        plt.plot(fine_frequency, self.GP_liklehood, '-', color='black')
        plt.xlabel('Frequency [Hz]', fontsize=14, color='black')
        plt.ylabel('Log Likelihood', fontsize=14, color='black')
        plt.title('')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    '''
        Following are visualisation Tools With Data and Priors
    '''

    def _r(self, t, *r_values, w, phi, b):

        # Special Function to make r vector on demand, See Gregory Paper for r vector
        index = self.j(t,w, phi, b) - 1  # Subtract 1 to match Python's 0-indexing
        return r_values[index]   

    def Sample(self, frequency = None, bins = None):

        if frequency is None:
            frequency = (self.w_max + self.w_min)
        if bins is None:
            bins = self.m

        # Plotting Specified Model along with Data
        w = frequency
        phi = 0
        b = bins
        r_values = np.random.uniform(self.rmin, self.rmax, size = b)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.time, [self._r(time, *r_values, w = w , phi = phi, b = b) for time in self.time], color='orange', alpha = 0.6)
        net_bins = (max(self.data.T[0])*w*bins).astype(int)
        #verticle line for bins
        for i in range(net_bins+1):
            plt.vlines((i)*1/(w*bins), self.rmin, r_values[np.mod(i, bins)], color='orange', alpha = 0.6)
        #Verticle line for first bin
        for i in range(net_bins//bins +1):
            plt.vlines(i/w, self.rmin, self.rmax , colors='black')
        #verticle lines to enclose periodic set of bins
        plt.hlines(self.rmin, 0, max(self.data.T[0]), colors='black')
        plt.hlines(self.rmax, 0, max(self.data.T[0]), colors='black')
        #plt.vlines(1/(w*bins), 0, r_values[1], colors='red')
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')
        plt.xlabel('time')
        plt.ylabel('flux')
        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('')
        plt.show()

    def Sample_bin(self, frequency = None, bins = None ):

        if frequency is None:
            frequency = (self.w_max + self.w_min)
        if bins is None:
            bins = self.m

        # Ploting with Similar Bins Overlayed (Model and Data)
        w = frequency
        phi = 0
        b = bins
        r_values = np.random.uniform(self.rmin, self.rmax, size = b)
        fig = plt.figure(figsize=(12, 6))
        #time in first period
        first_period_time = np.linspace(0, 1/w, 5000)
        plt.plot(first_period_time, [self._r(time, *r_values, w = w , phi = phi, b = b) for time in first_period_time], color='orange')
        net_bins = (max(self.data.T[0])*w*bins).astype(int)
        #verticle line for bins
        for i in range(bins):
            plt.vlines((i)*1/(w*bins), self.rmin, r_values[i], colors = 'orange')
        #Verticle line for first bin
        plt.vlines(0, self.rmin , self.rmax  , colors='black')
        plt.vlines(1/w , self.rmin , self.rmax , colors='black')
        #verticle lines to enclose periodic set of bins
        plt.hlines(self.rmin, 0, 1/w, colors='black')
        plt.hlines(self.rmax, 0, 1/w, colors='black')
        #plt.vlines(1/(w*bins), 0, r_values[1], colors='red')
        plt.scatter(np.mod(self.data[:, 0], 1/w) , self.data[:, 1], color = 'red')
        plt.xlabel('time')
        plt.ylabel('flux')
        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('')
        plt.show()


    def Sample_polar(self, frequency = None, bins = None):

        #Note, Polar Plot can't represent negative height of bins
        if frequency is None:
            frequency = (self.w_max + self.w_min)
        if bins is None:
            bins = self.m

        # Polar Plot with Similar Bins Overlayed (Model and Data)   
        w = frequency
        phi = 0
        b = bins
        r_values = np.random.uniform(self.rmax/5, self.rmax, size = b)
        fig = plt.figure(figsize=(8, 8))
        x_axis =  np.mod(self.data[:, 0], 1/w)
        y_axis =  self.data[:, 1]
        theta = np.mod(self.data[:, 0], 1/w)*(2*np.pi)*w #scaled x_axis
        radius = y_axis
        #plt.vlines(1/(w*bins), 0, r_values[1], colors='red')
        plt.scatter( radius*np.cos(theta), radius*np.sin(theta), color = 'red')
        #time in first period
        first_period_time = np.linspace(0, 1/w, 10000)
        first_period_r = [self._r(time, *r_values, w = w , phi = phi, b = b) for time in first_period_time]
        polar_time = first_period_time * (2*np.pi/(max(np.linspace(0, 1/w, 250))))
        #plot bins height
        plt.plot(first_period_r*np.cos(polar_time), first_period_r*np.sin(polar_time), color='orange')
        #verticle line for bins
        for i in range(bins):
            x = i*np.pi*2/bins
            y = self.rmax
            plt.plot([0, y*np.cos(x)], [0, y*np.sin(x)], 'r-', color = 'black')
        #Plot outermost Circle
        plt.plot(self.rmax*np.cos(polar_time), (self.rmax)*np.sin(polar_time), color = 'black')
        #verticle lines to enclose periodic set of bins
        plt.xlim(-(self.rmax), (self.rmax))
        plt.ylim(-(self.rmax), (self.rmax))
        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    '''
        Miscellaneous Methods
    '''

    def GP_signal_plot(self):

        #reshape data for Gausian process 
        X_train = self.data.T[0].reshape(-1,1)
        Y_train = self.data.T[1].reshape(-1,1)
        min_period = 1/self.w_max
        max_period = 1/self.w_min
        kernel = ExpSineSquared(length_scale=1.0, periodicity= 0.4,
                                   
                                    periodicity_bounds=(min_period, max_period))  + WhiteKernel(noise_level=self.data.T[2][0])
        # Fit the Gaussian process regressor
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0).fit(X_train, Y_train)
        # Generate many points in the range of the training data
        X_ = np.linspace(X_train.min(), X_train.max(), 1000)[:, np.newaxis]
        # Predict the mean and standard deviation at those points
        y_mean, y_std = gp.predict(X_, return_std=True)
        fig = plt.figure(figsize=(12, 6))
        # Plot the training data
        plt.scatter(X_train, Y_train, c='k', label='Training data')
        # Plot the mean of the GP
        plt.plot(X_, y_mean, 'b-', label='Mean of GP')
        # Plot the standard deviation of the GP
        plt.fill_between(X_[:, 0], y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='Std. dev. of GP')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

class Gregory_functions:#This is Model Used to generate Bins Structure in GL Method
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


class gl_calculator(Gregory_functions):
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
        self.time = np.linspace(0, self.data[-1,0], 1000)#high resolution time for ploting only
        self.prow_w = []
        self.prob_m = []
        self.prob_avg_w = []

        #data and priors are dictionary specified when class is instantiated
        # m is max bins ,w_min/ w_max, r_min, r_max are specified in priors
        #W_values are discrete values of w where calculation is performed
        #b_values are diffrent model with number of bins where calculation is performed
        # Last three are class variable, which will be calculated when class methods are performed on data      

    def Pw_dm(self):

        # Integral over phi, will give normalised probability of frequency
        # Performed over all w_values and b_values
        self.prob_w = [
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
                    epsabs=1.0e-1
                )[0]
            ) for b in self.b_values
        ]

    def plot_Pw_m(self,specified_bin):

        #Ploting probability distribution of frequency(Unormalised) for a specific model. Argument take number of bins of model.
        #Cal be executed right after integral pw_dm, make sure calculation is done before executing             
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.w_values, (self.prob_w[specified_bin-2]), 'o-')
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
        pio = np.array([[self.prob_m[b-2] * x for x in res_row] for b, res_row in zip(self.b_values, self.prob_w)])  
        #pio = np.array(hio)
        kio = sum(pio[i-2] for i in self.b_values)
        nio = [x/sum(self.prob_m) for x in kio]
        self.prob_avg_w = nio
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.plot(self.w_values, self.prob_avg_w, 'o-')    
        plt.xlabel('W Values')
        plt.ylabel('Unnormalised Probability')
        plt.show()