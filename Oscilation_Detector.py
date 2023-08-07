import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
from GL_functions import GL_functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared



class Oscilation_Detector(GL_functions):
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
        self.freq = []
        self.power = []
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


    def Compute_GL(self):
        self.log_prob_m_matrix = [[np.log(self.liklihood(w, 1.3, b)[0]) for w in self.w_values]for b in self.b_values]

        self.GL = np.sum(np.array(self.log_prob_m_matrix), axis=0)

        #Save best frequency of GL
        self.GL_result = self.w_values[np.nanargmax(self.GL)]  #nanargmax will work because argmax can give nan values



    def Plot_GL(self):


        fig = plt.figure(figsize=(12, 6))

        # Plotting the data
        plt.plot(self.w_values, self.GL, color='black')

        # Adding labels and title
        plt.title('', fontsize=16, color='black')
        plt.xlabel('w_values', fontsize=14, color='black')
        plt.ylabel('Log Likelihood', fontsize=14, color='black')


        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.show()

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
           
    def plot_data(self):
        #Ploting the raw data, used for calculation
              
        fig = plt.figure(figsize=(12, 6))  # create a figure
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')
        plt.vlines(self.data[:, 0], 0, self.data[:, 1], linewidth=0.5)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.grid(True)
        
        plt.show()

    def Compute_FFT(self):
        dt = float(self.data[1:2,0]) - float(self.data[0:1,0])
        N = len(self.data[:,0])

        #Compute FFT
        F = np.fft.fft(self.data[:,1])
        F = np.abs(F[:N//2])  # take only the positive-frequency terms
        freq = np.fft.fftfreq(N, dt)[:N//2]  # compute the frequencies
        
        #Compute power spectral density
        power = np.abs(F)**2

        mask = (freq >= self.priors['w_min']) & (freq <= self.priors['w_max'])
        self.power_mask = power[mask]

        self.freq = freq
        self.power = power
        self.fft_result = self.freq[mask][np.nanargmax(self.power[mask])]

    def Plot_FFT(self):

        freq = np.array(self.freq)
        power = np.array(self.power)
        #plottting
        plt.figure(figsize=(12,6))
        plt.title('Frequency Power Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')

        #only plot the frequencies in range(w_min, w_max)
        mask = (freq >= self.priors['w_min']) & (freq <= self.priors['w_max'])

        plt.plot(freq[mask], power[mask])

        plt.show()

    def _r(self, t, *r_values, w, phi, b):

        index = self.j(t,w, phi, b) - 1  # Subtract 1 to match Python's 0-indexing
        return r_values[index]   

    def sample(self):

        w = np.random.uniform(self.priors['w_min'], self.priors['w_max'])
        phi = np.random.uniform(0, 2*np.pi)
        b = np.random.randint(2 , self.m)

        r_values = np.random.uniform(self.rmin, self.rmax, size = b)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.time, [self._r(time, *r_values, w = w , phi = phi, b = b) for time in self.time], color='black')
        plt.xlabel('time')
        plt.ylabel('flux')
        # Hiding the axis numbers
        plt.xticks([])
        plt.yticks([])

        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title('Sample GL Function')
        plt.show()


    def sample_bin(self, frequency, bins):

        w = frequency
        phi = 0
        b = bins

        r_values = np.random.uniform(self.rmin, self.rmax, size = b)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.time, [self._r(time, *r_values, w = w , phi = phi, b = b) for time in self.time], color='orange', alpha = 0.6)

        net_bins = (max(self.data.T[0])*w*bins).astype(int)

        #verticle line for bins
        for i in range(net_bins+1):
            plt.vlines((i)*1/(w*bins), 0, r_values[np.mod(i, bins)], color='orange', alpha = 0.6)

        #Verticle line for first bin
        for i in range(net_bins//bins +1):
            plt.vlines(i/w, 0, (self.rmax + self.rmax/5) , colors='black')

        #verticle lines to enclose periodic set of bins
        plt.hlines(0, 0, max(self.data.T[0]), colors='black')
        plt.hlines((self.rmax + self.rmax/5), 0, max(self.data.T[0]), colors='black')

        

        #plt.vlines(1/(w*bins), 0, r_values[1], colors='red')
        plt.scatter(self.data[:, 0], self.data[:, 1], color = 'red')

        plt.xlabel('time')
        plt.ylabel('flux')


        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title('Sample GL Function')
        plt.show()

    def sample_bin_overlay(self, frequency, bins):

        w = frequency
        phi = 0
        b = bins

        r_values = np.random.uniform(self.rmin, self.rmax, size = b)

        fig = plt.figure(figsize=(12, 6))

        #time in first period
        first_period_time = np.linspace(0, 1/w, 1000)
        plt.plot(first_period_time, [self._r(time, *r_values, w = w , phi = phi, b = b) for time in first_period_time], color='orange')

        net_bins = (max(self.data.T[0])*w*bins).astype(int)

        #verticle line for bins
        for i in range(bins):
            plt.vlines((i)*1/(w*bins), 0, r_values[i], colors = 'orange')

        #Verticle line for first bin
        plt.vlines(0, 0 , (self.rmax + self.rmax/5) , colors='black')
        plt.vlines(1/w , 0 , (self.rmax + self.rmax/5) , colors='black')

        #verticle lines to enclose periodic set of bins
        plt.hlines(0, 0, 1/w, colors='black')
        plt.hlines((self.rmax + self.rmax/5), 0, 1/w, colors='black')

        

        #plt.vlines(1/(w*bins), 0, r_values[1], colors='red')
        plt.scatter(np.mod(self.data[:, 0], 1/w) , self.data[:, 1], color = 'red')

        plt.xlabel('time')
        plt.ylabel('flux')


        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title('Sample GL Function')
        plt.show()


    def sample_bin_overlay_polar(self, frequency, bins):

        w = frequency
        phi = 0
        b = bins

        r_values = np.random.uniform(self.rmin, self.rmax, size = b)

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
            y = self.rmax + self.rmax/5
            plt.plot([0, y*np.cos(x)], [0, y*np.sin(x)], 'r-', color = 'black')

        #Plot outermost Circle
        plt.plot((self.rmax+self.rmax/5)*np.cos(polar_time), (self.rmax+self.rmax/5)*np.sin(polar_time), color = 'black')


        #verticle lines to enclose periodic set of bins
        plt.xlim(-(self.rmax+ self.rmax/4), (self.rmax+ self.rmax/4))
        plt.ylim(-(self.rmax+ self.rmax/4), (self.rmax+ self.rmax/4))
  

        # Optional: Adding a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title('Sample GL Function')
        plt.show()


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

    def Compute_GP(self,resolution):
            #reshape data for Gausian process 
            X_Train = self.data.T[0].reshape(-1,1)
            Y_Train = self.data.T[1].reshape(-1,1)


            # Initialize lists to store the periods and corresponding log-marginal likelihoods
            fine_periods = []
            fine_log_marginal_likelihoods = []

            # Loop over periods
            min_period = 1/self.priors['w_max']
            max_period = 1/self.priors['w_min']
            w_values_GP = np.arange(1/self.priors['w_max'], 1/self.priors['w_min'], (max_period-min_period)/resolution)
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

            # Plot the log-marginal likelihood as a function of the period
            #save best period as frequency
            self.GP_result = 1/fine_periods[np.nanargmax(np.array(fine_log_marginal_likelihoods))]
            self.GP_period = fine_periods
            self.GP_liklehood = fine_log_marginal_likelihoods
            
    
    

    def Plot_GP(self):
        

        # Plot the log-marginal likelihood as a function of the period

        fig = plt.figure(figsize=(12, 6))  # create a figure
        fine_frequency = 1/np.array(self.GP_period)
        plt.plot(fine_frequency, self.GP_liklehood, '-')
        plt.xlabel('Period')
        plt.ylabel('Log-Marginal Likelihood')
        plt.title('Log-Marginal Likelihood as a Function of the Period')
        plt.grid(True)
        plt.show()

    def GP_signal_plot(self):
                #reshape data for Gausian process 
        X_train = self.data.T[0].reshape(-1,1)
        Y_train = self.data.T[1].reshape(-1,1)

        min_period = 1/self.priors['w_max']
        max_period = 1/self.priors['w_min']

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

        
