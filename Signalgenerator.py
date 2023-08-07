import numpy as np
import pandas as pd
from scipy.signal import sawtooth
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from scipy.signal import sawtooth
import matplotlib.pyplot as plt

class SignalGenerator:
    '''
    Generate fake periodic signal
    Parameters of signal to be determined in dictionary when class is instantiated.
    Different types of periodic signal are methods of class.
    Generate pure class is special methods which plots data without sparsity and noise added for visualization.

    Generated data is of the shape of numpy array,
    with three columns (time, flux, noise)

    If one method is followed by another previous generated signal will be overwritten
    '''
    def __init__(self, time_range=100, no_data_points=100, sparcity=0, frequency=0.035, phase=0, amplitude=1, dc=4, noise=0.5, random_seed=41, asymmetry=0.5, sample_rate=1):
        self.time_range = time_range
        self.no_data_points = no_data_points
        self.sparcity = sparcity
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.dc = dc
        self.noise = noise
        self.random_seed = random_seed
        self.asymmetry = asymmetry
        self.sample_rate = sample_rate
        self.data = None
        self.flux = None
        self.time = np.linspace(0, self.time_range, self.no_data_points)*self.sample_rate
        self.time_p = np.linspace(0, self.time_range, 1000)*self.sample_rate

        # Generating sine signal at instantiation
        self.generate_sine_signal()

    def generate_sine_signal(self):
        np.random.seed(self.random_seed)
        amp = self.amplitude
        flux = amp*np.sin(2*np.pi*self.time * self.frequency + self.phase) + self.dc
        flux_p = amp*np.sin(2*np.pi*self.time_p * self.frequency + self.phase) + self.dc
        self.flux = flux_p
        noise = np.random.normal(0, self.noise, self.no_data_points)
        flux_noisy = flux + noise
        df = pd.DataFrame({'time': self.time, 'flux': flux_noisy, 'error': self.noise})
        num_rows_to_delete = int(len(df) * self.sparcity / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def generate_triangular_signal(self):
        np.random.seed(self.random_seed)
        amp = self.amplitude
        flux = amp*sawtooth(2*np.pi*self.time * self.frequency + self.phase, self.asymmetry) + self.dc
        flux_p = amp*sawtooth(2*np.pi*self.time_p * self.frequency + self.phase, self.asymmetry) + self.dc
        self.flux = flux_p
        noise = np.random.normal(0, self.noise, self.no_data_points)
        flux_noisy = flux + noise
        df = pd.DataFrame({'time': self.time, 'flux': flux_noisy, 'error': self.noise})
        num_rows_to_delete = int(len(df) * self.sparcity / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def generate_impulsive_signal(self):
        np.random.seed(self.random_seed)
        amp = self.amplitude
        period = 1/self.frequency
        max_val = amp
        decay_rate = self.asymmetry
        flux = max_val * np.exp(-decay_rate * (self.time % period)) + self.dc
        flux_p = max_val * np.exp(-decay_rate * (self.time_p % period)) + self.dc
        self.flux = flux_p
        noise = np.random.normal(0, self.noise, self.no_data_points)
        flux_noisy = flux + noise
        df = pd.DataFrame({'time': self.time, 'flux': flux_noisy, 'error': self.noise})
        num_rows_to_delete = int(len(df) * self.sparcity / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def plot_pure_data(self):
        if self.data is None:
            print("No data available. Generate a signal first.")
            return
        fig = plt.figure(figsize=(12, 6))
        plt.plot(self.time_p, self.flux)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Time Series Data')
        plt.show()

    def plot_noise_data(self):
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
