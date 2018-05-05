import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import predict, update, KalmanFilter

class Filter():
    def __init__(self, x0):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Initial state estimate
        self.kf.x = np.array([x0, 0])
        # Initial Covariance matrix
        self.kf.P = np.eye(2) * 2**10

        # State transition function
        self.kf.F = np.array([[1., 1], [0., 1.]])
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=1)
        # Measurement noise
        self.kf.R = np.array([[50]])
        # Measurement function
        self.kf.H = np.array([[1., 0.]])
        # likelihood
        
    def likelihood(self, z):
        return self.kf.likelihood(z)
        
    def run(self, z):
        self.kf.predict()
        self.kf.update(z)  
        
    def position(self):
        return self.kf.x[0]