import numpy as np

def cw(t, s, omega): 
    r = s[0:3] 
    r_dot = s[3:6]
    r_ddot = np.array([2 * omega * r_dot[2  ], 
                        - omega ** 2 * r[1], 
                        3 * omega ** 2 * r[2] - 2 * omega * r_dot[0]])

    return np.concatenate([r_dot, r_ddot])


