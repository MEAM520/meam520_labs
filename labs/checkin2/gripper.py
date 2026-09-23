"""
Date: 09/23/2026

Purpose: This script checks the gripper state, opens and closes it,
and commands the gripper to specified configurations.

Try changing the target command to see what the gripper does!

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

### ADD STUDENT CODE HERE ###

#arm.exec_gripper_cmd(0.1)