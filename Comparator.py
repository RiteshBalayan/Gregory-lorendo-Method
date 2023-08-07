from Oscilation_Detector import Oscilation_Detector
from SignalGenerator import Signalgenerator

class Comparator:
    def __init__(self, Data_generator, Priors):
        self.Data_generator = Data_generator
        self.Priors = Priors
        self.data = []

    def Get_timeseries(self, type):
        dat = Signalgenerator(Data_generator)
        if type == sine :
            self.data = dat.generate_sine_signal()
        elif type == Triangle :
            self.data = dat.generate_triangular_signal()
        elif type == impulse :
            self.data = dat.generate_impulsive_signal()

    def Run_experiments(self, num_time, type, FFT = True, GL = True, GP = True, ):
        self.data = self.Get_timeseries(type) 
        pc = Oscilation_Detector(self.data, self.Priors)
        
        pc.Compute_FFT()
        pc.Compute_GL()
        pc.



        

    
