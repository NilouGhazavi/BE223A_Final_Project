#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:51:47 2024

@author: niloughazavi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = np.loadtxt('/Users/niloughazavi/Desktop/Desktop-Nilou/Nilou/PhD_Courses/BIOENGR223A/Project/Motion_5k_frames.csv', delimiter=',')

#  [x_shift, y_shift]
x_shift = data[:, 0]
y_shift = data[:, 1]



# periodic motion due to breathing , heartbeat, cyclical movement 
# each point : magnitude of shift of each frame
plt.plot(x_shift)
plt.plot(y_shift)
plt.xlabel('Frame')
plt.ylabel('shift')


# 
# Plot histogram and Q-Q plot
def plot_distribution(data, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(data, bins=50, density=True, alpha=0.7, color='skyblue')
    ax1.set_title(f'{title} - Histogram')
    ax1.set_xlabel('Shift (pixels)')
    ax1.set_ylabel('Density')
    
    
    mu, std = stats.norm.fit(data)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p, 'k', linewidth=2)
    ax1.text(0.05, 0.95, f'μ = {mu:.2f}\nσ = {std:.2f}', transform=ax1.transAxes, 
             verticalalignment='top')
    
    # Q-Q plot (Quantile-Quantile plot): compare the data to the theoretical normal distribution
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# Analyze X and Y shifts
plot_distribution(x_shift, 'X Shift')
plot_distribution(y_shift, 'Y Shift')

# Perform Shapiro-Wilk test for normality

# Null Hypothesis : Both x and y shifts are normally distributed 
# Alternative Hypothesis: Data is not normally distributed 

def test_normality(data, name):
    stat, p = stats.shapiro(data)
    print(f'{name} - Shapiro-Wilk test:')
    print(f'Statistic: {stat}, p-value: {p}')
    print(f'The data is {"likely normal" if p > 0.05 else "not normal"} (α = 0.05)\n')

test_normality(x_shift, 'X Shift')
test_normality(y_shift, 'Y Shift')

# Calculate and print correlation between X and Y shifts
correlation = np.corrcoef(x_shift, y_shift)[0, 1]
print(f'Correlation between X and Y shifts: {correlation:.4f}')





# generate synthetic data 
def generate_synthetic_data(num_frames, noise_params):
    t = np.arange(num_frames)
    
    # Generate base motion 
    x_base = 2 * np.sin(2 * np.pi * 0.01 * t)
    y_base = 1.5 * np.cos(2 * np.pi * 0.015 * t)
    
    # Add non-Gaussian noise (laplace)
    x_noise = np.random.laplace(0, noise_params['scale'], num_frames)
    y_noise = np.random.laplace(0, noise_params['scale'], num_frames)
    
    x_motion_base = x_base 
    y_motion_base = y_base 
    x_motion_noise= x_base + x_noise
    y_motion_noise = y_base + y_noise
    
    return np.column_stack((x_motion_base, y_motion_base)), np.column_stack((x_motion_noise, y_motion_noise))




num_frames = 1000

# scale parameter shows the spread or dispersion of the distribution
noise_params = {'scale': 0.5}
synthetic_motion_base, synthetic_motion_noise = generate_synthetic_data(num_frames, noise_params)

x_shift_syn_b = synthetic_motion_base[:, 0]
y_shift_syn_b = synthetic_motion_base[:, 1]


plt.plot(x_shift_syn_b)
plt.plot(y_shift_syn_b)
plt.xlabel('Frame')
plt.ylabel('shift')

x_shift_syn_n = synthetic_motion_noise[:, 0]
y_shift_syn_n = synthetic_motion_noise[:, 1]

plt.plot(x_shift_syn_n)
plt.plot(y_shift_syn_n)
plt.xlabel('Frame')
plt.ylabel('shift')



true_motion=synthetic_motion_base
observed_motion=synthetic_motion_noise




##

# Kalman filter assumes that the noise is normal distribution 






# Kalman Filter 
class RobustKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x) * 1000  # Higher initial uncertainty
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)

    def initialize(self, initial_state):
        self.x[:self.dim_z] = initial_state.reshape(self.dim_z, 1)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        z = z.reshape(self.dim_z, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Robust update using Huber loss
        threshold = 1.345  # Tunable parameter
        normalized_innovation = np.dot(np.linalg.inv(S), y)
        scale = np.minimum(1, threshold / np.linalg.norm(normalized_innovation))
        
        self.x = self.x + np.dot(K, scale * y)
        self.P = self.P - np.dot(np.dot(K, scale * self.H), self.P)

    def set_parameters(self, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R



def evaluate_kalman_filter(kf, true_motion, measurements):
    mse_list = []
    estimated_motion = []
    for i in range(len(true_motion)):
        kf.predict()
        kf.update(measurements[i])
        
        estimated_position = kf.x[:2, 0]
        estimated_motion.append(estimated_position)
        
        error = np.mean((true_motion[i] - estimated_position)**2)
        mse_list.append(error)
    
    return np.array(mse_list), np.array(estimated_motion)


# Set up Kalman Filter
kf = RobustKalmanFilter(dim_x=4, dim_z=2)
kf.initialize(observed_motion[0])

dt = 1.0
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
Q = np.eye(4) * 0.01  # Reduced process noise
R = np.eye(2) * (noise_params['scale']**2 * 2)  # Increased measurement noise

kf.set_parameters(F, H, Q, R)

# Evaluate Kalman Filter
mse_list, estimated_motion = evaluate_kalman_filter(kf, true_motion, observed_motion)
print(f"Mean Squared Error: {np.mean(mse_list)}")

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Motion plot
ax1.plot(true_motion[:, 0], label='True X')
ax1.plot(true_motion[:, 1], label='True Y')
ax1.plot(observed_motion[:, 0], 'o', alpha=0.5, markersize=2, label='Observed X')
ax1.plot(observed_motion[:, 1], 'o', alpha=0.5, markersize=2, label='Observed Y')
ax1.plot(estimated_motion[:, 0], '--', label='Estimated X')
ax1.plot(estimated_motion[:, 1], '--', label='Estimated Y')
ax1.set_ylabel('Shift (pixels)')
ax1.set_title('Kalman Filter Performance on Synthetic Data')
ax1.legend()

# Error plot
ax2.plot(mse_list, label='MSE')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Error Through Correction')
ax2.legend()

plt.tight_layout()
plt.show()

# Calculate and print error statistics
print(f"Final MSE: {mse_list[-1]}")
print(f"Max MSE: {np.max(mse_list)}")
print(f"Min MSE: {np.min(mse_list)}")


