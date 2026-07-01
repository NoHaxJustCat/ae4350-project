import libs
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

earth_mu = 3.986 * 10 ** 14 # m ** 3 / s ** 2
radius = (6378 + 600) * 10 ** 3 # m

omega = np.sqrt(earth_mu / radius ** 3)

state_i = np.array([5, 0, 5, 0, 0, 0])

T = 2 * np.pi/omega

t_span = (0, 2*T)   
t_eval = np.linspace(*t_span, 1000)

sol = integrate.solve_ivp(libs.cw, t_span, state_i,
                           rtol=1e-12, atol=1e-12,
                           args=(omega,), t_eval=t_eval)


plt.plot(sol.y[0, :], sol.y[2, :])   # x (V-bar) vs z (R-bar) is the classic CW ellipse view
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.axis('equal')
plt.grid(True)
plt.show()