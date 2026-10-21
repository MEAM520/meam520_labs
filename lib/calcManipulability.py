import numpy as np
from lib.calcJacobian import calcJacobian

def calcManipulability(q_in):
    J = calcJacobian(q_in)

    J_pos = J[:3,:]
    M = J_pos @ J_pos.T

    mu = 0.0
    ## STUDENT CODE HERE
    # Calculate mu

    
    ## END STUDENT CODE
    
    return mu, M


if __name__ == '__main__':
    # q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    print(calcManipulability(q))
