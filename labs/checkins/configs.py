"""
Date: 09/09/2026

Purpose: This script creates an ArmController and uses it to command the arm's
joint positions to a series of configurations.

Try changing the target positions to see what the arm does!

"""
import sys
import rospy
import numpy as np
from math import pi

from core.interfaces import ArmController

rospy.init_node('configs')

arm = ArmController()

neutral_q = arm.neutral_position()
arm.safe_move_to_position(neutral_q)

print("Joint Limits")
print(arm.joint_limits())
print("Current Joint Position")
print(arm.get_positions())
print("\n")

### ADD STUDENT CODE HERE ###

q1 = np.array([0,-1,0,-2,0,1,1]) # TODO: try changing this!
arm.safe_move_to_position(q1)

rospy.sleep(3.0)

q2 = np.array([0,-1,0,-2,0,1,1]) # TODO: try changing this!
arm.safe_move_to_position(q2)

rospy.sleep(3.0)

q3 = np.array([0,-1,0,-2,0,1,1]) # TODO: try changing this!
arm.safe_move_to_position(q3)
