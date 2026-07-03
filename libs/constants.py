import numpy as np

earth_mu = 3.986 * 10 ** 14 # m ** 3 / s ** 2
radius = (6378 + 600) * 10 ** 3 # m
omega = np.sqrt(earth_mu / radius ** 3)
T = 2 * np.pi/omega

gamma = 0.95  # Discount rate [7]
num_episodes = 5000
max_steps = 50