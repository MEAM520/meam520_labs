"""
Date: 08/26/2021

Purpose: This script creates an ArmController and uses it to command the arm's
joint positions and gripper.

Try changing the target position to see what the arm does!

"""

import sys
import rospy
import numpy as np
from math import pi

from core.interfaces import ArmController

rospy.init_node('demo')

arm1 = ArmController(id=1)
arm1.set_arm_speed(0.2)
arm2 = ArmController(id=2)
arm2.set_arm_speed(0.2)

arm1.close_gripper()
arm2.close_gripper()

q = arm1.neutral_position()
arm1.safe_move_to_position(q)
arm1.open_gripper()
arm2.safe_move_to_position(q)
arm2.open_gripper()

q = np.array([0,-1 ,0,-2,0,1,1]) # TODO: try changing this!
arm1.safe_move_to_position(q)
arm1.close_gripper()
arm2.safe_move_to_position(q)
arm2.close_gripper()
