import numpy as np
import pandas as pd
from scipy.signal import sawtooth
import matplotlib.pyplot as plt


class SignalGenerator:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.data = None
        self.flux = None

    def generate_sine_signal(self):
        np.random.seed(self.data_generator['Random_seed'])
        time = np.linspace(1, self.data_generator['time_range'], self.data_generator['no_data_points'])
        time_p = np.linspace(1, self.data_generator['time_range'], 1000)
        amp = self.data_generator['Amplitude']
        flux = amp*np.sin(time * self.data_generator['frequency'] + self.data_generator['Phase']) + self.data_generator['DC']
        flux_p = amp*np.sin(time_p * self.data_generator['frequency'] + self.data_generator['Phase']) + self.data_generator['DC']
        self.flux = flux_p
        noise = np.random.normal(0, self.data_generator['Noise'], self.data_generator['no_data_points'])
        flux_noisy = flux + noise
        df = pd.DataFrame({'time': time, 'flux': flux_noisy, 'error': self.data_generator['Noise']})
        num_rows_to_delete = int(len(df) * self.data_generator['sparcity'] / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def generate_triangular_signal(self):
        np.random.seed(self.data_generator['Random_seed'])
        time = np.linspace(1, self.data_generator['time_range'], self.data_generator['no_data_points'])
        time_p = np.linspace(1, self.data_generator['time_range'], 1000)
        amp = self.data_generator['Amplitude']
        flux = (
            amp*sawtooth(time * self.data_generator['frequency'] + self.data_generator['Phase'],
                     self.data_generator['asymetry']) +
            self.data_generator['DC']
        )
        flux_p = (
            amp*sawtooth(time_p * self.data_generator['frequency'] + self.data_generator['Phase'],
                     self.data_generator['asymetry']) +
            self.data_generator['DC']
        )
        self.flux = flux_p
        noise = np.random.normal(0, self.data_generator['Noise'], self.data_generator['no_data_points'])
        flux_noisy = flux + noise
        df = pd.DataFrame({'time': time, 'flux': flux_noisy, 'error': self.data_generator['Noise']})
        num_rows_to_delete = int(len(df) * self.data_generator['sparcity'] / 100)
        indices_to_delete = np.random.choice(df.index, size=num_rows_to_delete, replace=False)
        df = df.drop(indices_to_delete)
        self.data = np.column_stack((df['time'], df['flux'], df['error']))
        return self.data

    def plot_pure_data(self):
        if self.data is None:
            print("No data available. Generate a signal first.")
            return
        time_p = np.linspace(1, self.data_generator['time_range'], 1000)
        flux = self.flux
        plt.plot(time_p, flux)
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
        plt.plot(time, flux)
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.title('Time Series Data')
        plt.show()