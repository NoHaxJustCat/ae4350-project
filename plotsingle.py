from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def cw(t, s, omega): 
    r = s[0:3] 
    r_dot = s[3:6]
    r_ddot = np.array([2 * omega * r_dot[2  ], 
                        - omega ** 2 * r[1], 
                        3 * omega ** 2 * r[2] - 2 * omega * r_dot[0]])

    return np.concatenate([r_dot, r_ddot])


earth_mu = 3.986 * 10 ** 14 # m ** 3 / s ** 2
radius = (6378 + 600) * 10 ** 3 # m
omega = np.sqrt(earth_mu / radius ** 3)

state_i = np.array([5, 0, 5, 0, 0, 0])

T = 2 * np.pi/omega

t_span = (0, 2*T)   
t_eval = np.linspace(*t_span, 1000)

sol = integrate.solve_ivp(functions.cw, t_span, state_i,
                           rtol=1e-12, atol=1e-12,
                           args=(omega,), t_eval=t_eval)


plt.plot(sol.y[0, :], sol.y[2, :])
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.axis('equal')
plt.grid(True)
plt.show()