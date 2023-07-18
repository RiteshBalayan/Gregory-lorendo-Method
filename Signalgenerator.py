import numpy as np
import pandas as pd
from scipy.signal import sawtooth
import matplotlib.pyplot as plt


class SignalGenerator:
    '''
    Genetate fake periodic signal
    Parameters of signal to be determined in dictionary when class is instantiated.
    Diffrent types of periodic signal are methods of class.
    Generate pure class is special methods which plots data without sparcity and noise added for visualisation.

    Generated data is of the shape of numpy array,
    with three columns (time, flux, noise)

    If one methods is followed by another previous generated signal will be overwritten


    '''
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.data = None
        self.flux = None
        self.time = np.linspace(0, self.data_generator['time_range'], self.data_generator['no_data_points'])#time with asked resolution
        self.time_p = np.linspace(0, self.data_generator['time_range'], 1000)#high resolution time for ploting only
        '''
        fill entries of folling dictionary before instantiating:

        Data_generator = {
                          'time_range': , 'no_data_points': , 'sparcity': , 
                          'frequency': , 'Phase': , 'Amplitude': , 'DC':  ,
                          'Noise': , 'Random_seed': , 'asymetry' :    
                        }

        '''

    def generate_sine_signal(self):
        #generate one pure signal and one noisy signal

        np.random.seed(self.data_generator['Random_seed']) #use random seed

        amp = self.data_generator['Amplitude'] #use amplitude specefied

        flux = amp*np.sin(2*np.pi*self.time * self.data_generator['frequency'] + self.data_generator['Phase']) + self.data_generator['DC']
        flux_p = amp*np.sin(2*np.pi*self.time_p * self.data_generator['frequency'] + self.data_generator['Phase']) + self.data_generator['DC']
        #Requested flux and flux_p for ploting noisy free

        self.flux = flux_p # flux requested
        # add noise to flux
        noise = np.random.normal(0, self.data_generator['Noise'], self.data_generator['no_data_points'])
        flux_noisy = flux + noise

        # make data frame requested
        df = pd.DataFrame({'time': self.time, 'flux': flux_noisy, 'error': self.data_generator['Noise']})

        # make data-set sparse
        num_rows_to_delete = int(len(df) * self.data_generator['sparcity'] / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)

        #convert to numpy array
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data



    def generate_triangular_signal(self):
        #generate one pure signal and one noisy signal

        np.random.seed(self.data_generator['Random_seed'])#use random seed

        amp = self.data_generator['Amplitude']#use amplitude specefied

        flux = (
                     amp*sawtooth(self.time * self.data_generator['frequency'] + self.data_generator['Phase'],
                     self.data_generator['asymetry']) +
                     self.data_generator['DC']
        )
        flux_p = (
            amp*sawtooth(self.time_p * self.data_generator['frequency'] + self.data_generator['Phase'],
                     self.data_generator['asymetry']) +
            self.data_generator['DC']
        )#Requested flux and flux_p for ploting noisy free

        self.flux = flux_p # flux requested
        # add noise to flux
        noise = np.random.normal(0, self.data_generator['Noise'], self.data_generator['no_data_points'])
        flux_noisy = flux + noise

        # make data frame requested
        df = pd.DataFrame({'time': self.time, 'flux': flux_noisy, 'error': self.data_generator['Noise']})

        # make data-set sparse        
        num_rows_to_delete = int(len(df) * self.data_generator['sparcity'] / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)

        #convert to numpy array      
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def plot_pure_data(self):
        #plot pure data, noise free, uniformed sampled
        if self.data is None:
            print("No data available. Generate a signal first.")
            return
        #time_p = np.linspace(0, self.data_generator['time_range'], 1000)
        flux = self.flux
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.time_p, self.flux)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Time Series Data')
        plt.show()
        
    def plot_noise_data(self):
        #Plot requested data
        if self.data is None:
            print("No data available. Generate a signal first.")
            return
        time = self.data[:, 0]
        flux = self.data[:, 1]
        fig = plt.figure(figsize=(12, 6))
        plt.plot(time, flux)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Time Series Data')
        plt.show()