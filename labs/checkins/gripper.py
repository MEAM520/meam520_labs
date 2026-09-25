"""
Date: 09/24/2026

Purpose: This script checks the gripper state, opens and closes it. 

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

print(arm.get_gripper_state())
arm.open_gripper()
print(arm.get_gripper_state())
rospy.sleep(5)
arm.close_gripper()
print(arm.get_gripper_state())
