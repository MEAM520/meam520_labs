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
arm.close_gripper()
print(arm.get_gripper_state())
rospy.sleep(5)
arm.open_gripper()
print(arm.get_gripper_state())

lowered_q = np.array([0,-1,0,-2,0,1,1]) # TODO: change this configuration to lower the arm to a position where the gripper is near the table
arm.safe_move_to_position(lowered_q)
rospy.sleep(3)

### ADD STUDENT CODE HERE ###

#arm.exec_gripper_cmd(0.1)

### END STUDENT CODE ###

rospy.sleep(3)
raised_q = np.array([0,-1,0,-2,0,1,1]) # TODO: change this configuration to raise the arm to a position where the gripper is away from the table
arm.safe_move_to_position(raised_q)

rospy.sleep(5)

arm.safe_move_to_position(lowered_q)
arm.open_gripper()

rospy.sleep(5)
arm.safe_move_to_position(neutral_q)