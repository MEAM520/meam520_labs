import numpy as np
from lib.calcJacobian import calcJacobian

def calcManipulability(q_in):
    """
    Helper function for calculating manipulability ellipsoid and index

    INPUTS:
    q_in - 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]

    OUTPUTS:
    mu - a float scalar, manipulability index
    M  - 3 x 3 manipulability matrix for the linear portion
    """
    J = calcJacobian(q_in)

    J_pos = J[:3,:]
    M = J_pos @ J_pos.T

    ## STUDENT CODE STARTS HERE for the mu index, Hint: np.linalg.svd
    mu = 0.0

    return mu, M