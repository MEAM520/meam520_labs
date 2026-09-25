"""
Date: 09/24/2026

Purpose: This script checks uses the gripper to pick up and object and place it back down. 

Try playing with different grasping parameters!

"""
import sys
import rospy
import numpy as np
from math import pi

from core.interfaces import ArmController

rospy.init_node('gripper')

arm = ArmController()

neutral_q = arm.neutral_position()
arm.safe_move_to_position(neutral_q)

pregrasp_q = np.array([-0.01992212, -0.35365552, -0.04928855, -2.46110433,  0.02991587,  2.11028957, 0.76457473])
grasp_q = np.array([-0.01841442, -0.26857176, -0.04476331, -2.58254501,  0.0304124, 2.29239265, 0.74858136])
 

# Move to pregrasping configuration
arm.safe_move_to_position(pregrasp_q)

# Open the gripper 
arm.open_gripper()

# Move to the grasping configuration
arm.safe_move_to_position(grasp_q)

# Actuate the gripper

####### STUDENT CODE : EDIT the inputs to this function. 
####### DO NOT MODIFY ANYTHING ELSE
arm.exec_gripper_cmd(0.01)

# Move back to the pregrasping configuration
arm.safe_move_to_position(pregrasp_q)

# Pause, holding the object aloft
rospy.sleep(5)  # 5 seconds

# Move back to grasping configuration
arm.safe_move_to_position(grasp_q)

# Open the gripper, releasing the object
arm.open_gripper()

# Move back to neutral configuration
arm.safe_move_to_position(neutral_q)
