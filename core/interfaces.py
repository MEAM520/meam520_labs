#!/usr/bin/env python3
#
# MEAM 520 Arm Controller, Fall 2021
#
# *** MEAM520 STUDENTS SHOULD NOT MODIFY THIS FILE ***
#
# This code is *HEAVILY* based on / directly modified from the PandaRobot
# package authored by Saif Sidhik. The license and attribution of the original
# open-source package is below.
#
# Copyright (c) 2019-2021, Saif Sidhik
# Copyright (c) 2013-2014, Rethink Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# **************************************************************************/

"""
	@info
		Interface class for the Franka Robot in hardware and simulation

"""

import os
import copy
from dataclasses import is_dataclass
import rospy
import logging
import argparse
import numpy as np
import franka_interface
import itertools
from core.safety import Safety
from core.utils import time_in_seconds
from franka_core_msgs.msg import JointCommand
from meam520_labs.msg import BlockDetection, BlockDetectionArray

from gazebo_msgs.msg import LinkStates, ModelStates
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_matrix
import std_msgs.msg
from math import acos, sin, cos, pi

from core.utils import time_in_seconds, transform
from scipy.spatial.transform import Rotation as R
import tf
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
import math
from math import sin, cos, pi
from copy import deepcopy
from cv_bridge import CvBridge
import cv2

class ObjectDetector:       

	def __init__(self):
		self.team = rospy.get_param("team") # 'red' or 'blue'
		self.table_distance = 0.99
		self.block_size = 0.05 # in meters

		## class variables here
		self.image = None
		self.model = PinholeCameraModel()
		self.got_camera_info = False
		self.listener = tf.TransformListener()
		self.detections = []
		self.frame_height = self.frame_width = None

		self.mid_depth = None
		self.mid_rgb = None
		
		## all subscribers here
		self.gazebo_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_cb, queue_size=1)
		self.camera_hardware_sub = rospy.Subscriber('/block_detections', BlockDetectionArray, self.vision_cb, queue_size=1)
		self.camera_software_sub = rospy.Subscriber('/image_raw', Image, self.image_callback, queue_size=1)
		self.camera_info_sub = rospy.Subscriber('/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)

		# middle camera
		self.mid_depth_sub = rospy.Subscriber("/depth/depth/image_raw", Image, self.mid_depth_callback, queue_size=1)
		self.mid_rgb_sub = rospy.Subscriber("/depth/color/image_raw", Image, self.mid_rgb_callback, queue_size=1)

		# Transform from camera to end-effector
		if rospy.search_param('/use_sim_time'):
			self.H_ee_camera = np.array([[0, -1, 0, 5e-2],[1, 0, 0, 0], [0, 0, 1, -6e-2], [0, 0, 0, 1]])
		else:

			if self.team == 'red':
				self.H_ee_camera = np.array([[0, -1, 0, 3.00e-2],[1, 0, 0, 0.14e-2], [0, 0, 1, -5.224e-2], [0, 0, 0, 1]])
			elif self.team == 'blue':
				self.H_ee_camera = np.array([[0, -1, 0, 6.008e-2],[1, 0, 0, -1.5e-2], [0, 0, 1, -6.837e-2], [0, 0, 0, 1]])


	################
	## SIMULATION ##
	################
				
	def mid_depth_callback(self, msg):
		br = CvBridge()
		self.mid_depth = br.imgmsg_to_cv2(msg)

	def mid_rgb_callback(self, msg):
		br = CvBridge()
		self.mid_rgb = br.imgmsg_to_cv2(msg, desired_encoding="bgr8")

	def camera_info_callback(self, info):
		if self.got_camera_info == False:
			self.model.fromCameraInfo(info)
			self.frame_height = info.height
			self.frame_width = info.width
			self.got_info = True

	def image_callback(self, image):
		self.image = deepcopy(image)

	def gazebo_cb(self,msg):
		try:
			(trans,rot) = self.listener.lookupTransform('world', 'endeffector',rospy.Time(0))
			H_base_ee = R.from_quat(rot).as_matrix()
			H_base_ee = np.append(H_base_ee, np.array(trans).reshape(3,1), axis=1)
			H_base_ee = np.append(H_base_ee, np.array([0,0,0,1]).reshape(1,4), axis=0)
			H_base_camera = H_base_ee @ self.H_ee_camera

			blocks = []
			for (name,pose) in zip(msg.name,msg.pose):
				if 'cube' in name: 					
					H_world_block = self.pose_to_transform(pose)
					H_base_block = H_world_block.copy()
					if self.team == 'red':
						H_base_block[1,3] += (self.table_distance)
					elif self.team == 'blue':
						H_base_block[1,3] += (-self.table_distance)


					H_camera_base = np.linalg.inv(H_base_camera)
					H_camera_block = H_camera_base @ H_base_block
					block = (name, H_camera_block)
					if self.check_block_visibility(block):
						blocks.append(block)
			
			self.detections = blocks
		
		except Exception as e:
			pass

	def check_block_visibility(self,block):
		if self.got_info:
			block_position = block[1][:3,3]

			## check if the block is behind the camera
			if block_position[2] < 0.:
				return False

			roll, pitch, yaw = R.from_matrix(block[1][:3, :3]).as_euler('xyz')
			block_side1 = [block_position[0] + self.block_size/2 * cos(yaw), block_position[1] + self.block_size/2 * sin(yaw), block_position[2]]
			block_side2 = [block_position[0] + self.block_size/2 * cos(yaw+pi/2), block_position[1] + self.block_size/2 * sin(yaw+pi/2), block_position[2]]
			block_side3 = [block_position[0] + self.block_size/2 * cos(yaw+pi), block_position[1] + self.block_size/2 * sin(yaw+pi), block_position[2]]
			block_side4 = [block_position[0] + self.block_size/2 * cos(yaw+3*pi/2), block_position[1] + self.block_size/2 * sin(yaw+3*pi/2), block_position[2]]
			u1, v1 = self.model.project3dToPixel(block_side1)
			if u1 < 0 or v1 < 0 or u1 > self.frame_width or v1 > self.frame_height :
				return False
			
			u2, v2 = self.model.project3dToPixel(block_side2)
			if u2 < 0 or v2 < 0 or u2 > self.frame_width or v2 > self.frame_height:
				return False
			
			
			u3, v3 = self.model.project3dToPixel(block_side3)
			if u3 < 0 or v3 < 0 or u3 > self.frame_width or v3 > self.frame_height:
				return False
			
			
			u4, v4 = self.model.project3dToPixel(block_side4)
			if u4 < 0 or v4 < 0 or u4 > self.frame_width or v4 > self.frame_height:
				return False
			
			return True
		else:
			return False

	def pose_to_transform(self,pose):
		T = quaternion_matrix([
			pose.orientation.x,
			pose.orientation.y,
			pose.orientation.z,
			pose.orientation.w ])
		T[:3,3] = np.array([
			pose.position.x,
			pose.position.y,
			pose.position.z ])
		return T


	##############
	## HARDWARE ##
	##############


	def vision_cb(self, msg):
		blocks = []
		for i in range(len(msg.detections)):
			block_id = msg.detections[i].id
			is_dynamic = msg.detections[i].dynamic
			if is_dynamic:
				block_name = 'cube{}_dynamic'.format(block_id)
			else:
				block_name = 'cube{}_static'.format(block_id)
			position = msg.detections[i].pose.pose.position
			orientation_quat = msg.detections[i].pose.pose.orientation
			trans = [position.x, position.y, position.z]
			rot = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
			H_cam_block = R.from_quat(rot).as_matrix()
			H_cam_block = np.append(H_cam_block, np.array(trans).reshape(3,1), axis=1)
			H_cam_block = np.append(H_cam_block, np.array([0,0,0,1]).reshape(1,4), axis=0)
			blocks.append((block_name, H_cam_block))
		
		self.detections = blocks

	############
	## CLIENT ##
	############

	def get_detections(self):
		"""
		return the list of detections of blocks in camera frame

		output: tuple ( block_id, eg 'block3_static'
						Nx4x4, list of homogeneous transformation matrices
		"""
		return self.detections.copy()

	def get_H_ee_camera(self):
		"""
		return the transformation matrix Camera in end-effector frame

		output: 4x4 homogeneous transformation matrix
		"""
		try:
			(trans,rot) = self.listener.lookupTransform('panda_EE', 'camera',rospy.Time(0))
			self.H_ee_camera = R.from_quat(rot).as_matrix()
			self.H_ee_camera = np.append(self.H_ee_camera, np.array(trans).reshape(3,1), axis=1)
			self.H_ee_camera = np.append(self.H_ee_camera, np.array([0,0,0,1]).reshape(1,4), axis=0)
		
		except Exception as e:
			print(e)
			pass

		return self.H_ee_camera.copy()
	
	def get_mid_depth(self):
		return self.mid_depth
	
	def get_mid_rgb(self):
		return self.mid_rgb

class ArmController(franka_interface.ArmInterface):
	"""
		:bases: :py:class:`franka_interface.ArmInterface`

		:param on_state_callback: optional callback function to run on each state update

	"""
	#############################
	##                         ##
	##    "UNDER THE HOOD"     ##
	##      FUNCTIONALITY      ##
	##                         ##
	##   Students don't need   ##
	##   to use these methods  ##
	##                         ##
	#############################

	def __init__(self, on_state_callback=None):
		"""
			Constructor class.  Functions from `franka_interface.ArmInterface <https://justagist.github.io/franka_ros_interface/DOC.html#arminterface>`_

			:param on_state_callback: optional callback function to run on each state update
		"""

		self._logger = logging.getLogger(__name__)

		# ----- don't update robot state value in this class until robot is fully configured
		self._arm_configured = False

		# Parent constructor
		franka_interface.ArmInterface.__init__(self)

		self._jnt_limits = [{'lower': self.get_joint_limits().position_lower[i],
							 'upper': self.get_joint_limits().position_upper[i]}
							for i in range(len(self.joint_names()))]

		# number of joints
		self._nq = len(self._jnt_limits)
		# number of control commands
		self._nu = len(self._jnt_limits)

		self._configure(on_state_callback)

		self._tuck = [self._neutral_pose_joints[j] for j in self._joint_names]

		self._untuck = self._tuck

		self._q_mean = np.array(
			[0.5 * (limit['lower'] + limit['upper']) for limit in self._jnt_limits])

		self._franka_robot_enable_interface = franka_interface.RobotEnable(
			self._params)

		if not self._franka_robot_enable_interface.is_enabled():
			self._franka_robot_enable_interface.enable()

		self._time_now_old = time_in_seconds()

		self._arm_configured = True

		self.safe = Safety()


	def _configure(self, on_state_callback):

		if on_state_callback:
			self._on_state_callback = on_state_callback
		else:
			self._on_state_callback = lambda m: None

		self._configure_gripper(
			self.get_robot_params().get_gripper_joint_names())

		if self.get_robot_params()._in_sim:
			# Frames interface is not implemented for simulation controller
			self._frames_interface = None

	def _configure_gripper(self, gripper_joint_names):
		self._gripper = franka_interface.GripperInterface(
			ns=self._ns, gripper_joint_names=gripper_joint_names)
		if not self._gripper.exists:
			self._gripper = None
			return

	def _on_joint_states(self, msg):
		# Parent callback function is overriden to update robot state of this class

		franka_interface.ArmInterface._on_joint_states(self, msg)

		if self._arm_configured:
			self._state = self._update_state()
			self._on_state_callback(self._state)


	def _update_state(self):

		now = rospy.Time.now()

		state = {}
		state['position'] = self.get_positions()
		state['velocity'] = self.get_velocities()
		state['effort'] = self.get_torques()
		state['timestamp'] = {'secs': now.secs, 'nsecs': now.nsecs}
		state['gripper_state'] = self.get_gripper_state()

		return state

	def _format_command_with_limits(self, cmd):

		for (angle,limit,number) in zip(cmd,self.joint_limits(),range(7)):
			if angle < limit['lower'] or angle > limit['upper']:
				cmd[number] = min(limit['upper'],max(limit['lower'],angle))
				rospy.logwarn("Position {angle:2.2f} for joint {number} violates joint limits [{lower:2.5f},{upper:2.5f}]. Constraining within range.".format(
					number=number,
					lower=limit['lower'],
					upper=limit['upper'],
					angle=angle
				))

		return dict(zip(self.joint_names(), cmd[:7]))


	#####################
	##                 ##
	##  CONFIGURATION  ##
	##       AND       ##
	##   PARAMETERS    ##
	##                 ##
	#####################

	def joint_limits(self):
		"""
		:return: joint limits
		:rtype: [{'lower': float, 'upper': float}]
		"""
		return self._jnt_limits

	def q_mean(self):
		"""
		:return: mean of joint limits i.e. "center position"
		:rtype: [float]
		"""
		return self._q_mean

	def n_joints(self):
		"""
		:return: number of joints
		:rtype: int
		"""
		return self._nq

	def n_cmd(self):
		"""
		:return: number of control commands (normally same as number of joints)
		:rtype: int
		"""
		return self._nu

	def enable_robot(self):
		"""
			Re-enable robot if stopped due to collision or safety.
		"""
		self._franka_robot_enable_interface.enable()

	def set_arm_speed(self, speed):
		"""
		Set joint position speed (only effective for :py:meth:`move_to_joint_position`

		:type speed: float
		:param speed: ratio of maximum joint speed for execution; range = [0.0,1.0]
		"""
		self.set_joint_position_speed(speed)

	def set_gripper_speed(self, speed):
		"""
			Set velocity for gripper motion

			:param speed: speed ratio to set
			:type speed: float
		"""
		if self._gripper:
			self._gripper.set_velocity(speed)

	def neutral_position(self):
		return np.array(list(self._params.get_neutral_pose().values()))


	########################
	##                    ##
	##  FEEDBACK / STATE  ##
	##                    ##
	########################

	def get_positions(self, include_gripper=False):
		"""
		:return: current joint angle positions
		:rtype: [float]

		:param include_gripper: if True, append gripper joint positions to list
		:type include_gripper: bool
		"""
		joint_angles = self.joint_angles()

		joint_names = self.joint_names()

		all_angles = [joint_angles[n] for n in joint_names]

		if include_gripper and self._gripper:
			all_angles += self._gripper.joint_ordered_positions()

		return np.array(all_angles)

	def get_velocities(self, include_gripper=False):
		"""
		:return: current joint velocities
		:rtype: [float]

		:param include_gripper: if True, append gripper joint velocities to list
		:type include_gripper: bool
		"""
		joint_velocities = self.joint_velocities()

		joint_names = self.joint_names()

		all_velocities = [joint_velocities[n] for n in joint_names]

		if include_gripper and self._gripper:
			all_velocities += self._gripper.joint_ordered_velocities()

		return np.array(all_velocities)

	def get_torques(self, include_gripper=False):
		"""
		:return: current joint efforts (measured torques)
		:rtype: [float]

		:param include_gripper: if True, append gripper joint efforts to list
		:type include_gripper: bool
		"""
		joint_efforts = self.joint_efforts()

		joint_names = self.joint_names()

		all_efforts = [joint_efforts[n] for n in joint_names]

		if include_gripper and self._gripper:
			all_efforts += self._gripper.joint_ordered_efforts()

		return np.array(all_efforts)

	def get_gripper_state(self):
		"""
		Return just the Gripper state {'position', 'force'}.
		Only available if Franka gripper is connected.

		Note that the gripper has two jaws, so there are two position / force values.

		:rtype: dict ({str : numpy.ndarray (shape:(2,)), str : numpy.ndarray (shape:(2,))})
		:return: dict of position and force

		  - 'position': :py:obj:`numpy.ndarray`
		  - 'force': :py:obj:`numpy.ndarray`
		"""
		gripper_state = {}

		if self._gripper:
			gripper_state['position'] = self._gripper.joint_ordered_positions()
			gripper_state['force'] = self._gripper.joint_ordered_efforts()

		return gripper_state

	def get_state(self):
		"""
		Gets the full robot state including the gripper state and timestamp.
		See _update_state() above for fields.

		:return: robot state as a dictionary
		:rtype: dict {str: obj}
		"""
		return self._state

	#######################
	##                   ##
	##  MOTION COMMANDS  ##
	##                   ##
	#######################


	def move_to_position(self, joint_angles, timeout=10.0, threshold=0.00085, test=None, is_safe=False):
		"""
		Move to joint position specified (attempts to move with trajectory action client).
		This function will smoothly interpolate between the start and end positions
		in joint space, including ramping up and down the speed.

		This is a blocking call! Meaning your code will not proceed to the next instruction
		until the robot is within the threshold or the timeout is reached.

		.. note:: This method stops the currently active controller for trajectory tracking (and automatically restarts the controller(s) after execution of trajectory).

		:param joint_angles: desired joint positions, ordered from joint1 to joint7
		:type joint_angles: [float]
		:type timeout: float
		:param timeout: seconds to wait for move to finish [10]
		:type threshold: float
		:param threshold: position threshold in radians across each joint when
		 move is considered successful [0.00085]
		:param test: optional function returning True if motion must be aborted
		"""
		if is_safe==False:
			raise Exception("!!!++++++++++++++++++++++++++You are not using safe command!!!++++++++++++++++++++++++++++++++!!!")
		self.move_to_joint_positions(
			self._format_command_with_limits(joint_angles), timeout=timeout, threshold=threshold, test=test, use_moveit=False)

	def untuck(self):
		"""
		Move to neutral pose (using trajectory controller)
		"""
		self.move_to_position(self.neutral_position())

	def exec_gripper_cmd(self, pos, force=None):
		"""
		Move gripper joints to the desired width (space between finger joints), while applying
		the specified force (optional)

		:param pos: desired width [m]
		:param force: desired force to be applied on object [N]
		:type pos: float
		:type force: float

		:return: True if command was successful, False otherwise.
		:rtype: bool
		"""
		if self._gripper is None:
			return False

		width = min(self._gripper.MAX_WIDTH, max(self._gripper.MIN_WIDTH, pos))

		if force:
			holding_force = min(
				max(self._gripper.MIN_FORCE, force), self._gripper.MAX_FORCE)

			return self._gripper.grasp(width=width, force=holding_force)

		else:
			return self._gripper.move_joints(width)

	def open_gripper(self):
		"""
		Convenience function to open gripper all the way
		"""
		# behavior at true limit is unreliable
		self.exec_gripper_cmd(self._gripper.MAX_WIDTH * (1 - 1e-2) )

	def close_gripper(self):
		"""
		Convenience function to close gripper all the way
		Note: good grasping performance requires applying a force as well!
		"""
		self.exec_gripper_cmd(self._gripper.MIN_WIDTH)


	def exec_position_cmd(self, cmd):
		"""
		Execute position control on the robot (raw positions). Be careful while using. Send smooth
		commands (positions that are very small distance apart from current position).

		:param cmd: desired joint postions, ordered from joint1 to joint7
						(optionally, give desired gripper width as 8th element of list)
		:type cmd: [float]
		"""

		if len(cmd) > 7:
			gripper_cmd = cmd[7:]
			self.exec_gripper_cmd(*gripper_cmd)

		joint_command = self._format_command_with_limits(cmd)

		self.set_joint_positions(joint_command)

	def exec_velocity_cmd(self, cmd):
		"""
		Execute velocity command at joint level (using internal velocity controller)

		:param cmd: desired joint velocities, ordered from joint1 to joint7
		:type cmd: [float]
		"""
		joint_names = self.joint_names()

		velocity_command = dict(zip(joint_names, cmd))

		self.set_joint_velocities(velocity_command)

	def exec_torque_cmd(self, cmd):
		"""
		Execute torque command at joint level directly

		:param cmd: desired joint torques, ordered from joint1 to joint7
		:type cmd: [float]
		"""
		joint_names = self.joint_names()

		torque_command = dict(zip(joint_names, cmd))

		self.set_joint_torques(torque_command)


	def set_joint_positions_velocities(self, positions, velocities, is_safe=False):
		"""
		Commands the joints of this limb using specified positions and velocities using impedance control.
		Command at time t is computed as:

		:math:`u_t= coriolis\_factor * coriolis\_t + K\_p * (positions - curr\_positions) +  K\_d * (velocities - curr\_velocities)`


		:type positions: [float]
		:param positions: desired joint positions as an ordered list corresponding to joints given by self.joint_names()
		:type velocities: [float]
		:param velocities: desired joint velocities as an ordered list corresponding to joints given by self.joint_names()
		"""
		if is_safe==False:
			raise Exception("!!!++++++++++++++++++++++++++You are not using safe command!!!++++++++++++++++++++++++++++++++!!!")
		self._command_msg.names = self._joint_names
		self._command_msg.position = positions
		self._command_msg.velocity = velocities
		self._command_msg.mode = JointCommand.IMPEDANCE_MODE
		self._command_msg.header.stamp = rospy.Time.now()
		self._joint_command_publisher.publish(self._command_msg)


	def safe_move_to_position(self, joint_angles, timeout=10.0, threshold=0.00085, test=None):
		# The safety layer
		cur_safe = self.safe.test_new_configuration(joint_angles)

		if cur_safe == True:
			self.move_to_position(joint_angles, timeout, threshold, test, is_safe=True)
		else:
			print("Robot will hit the table!!! Aborting the current configuration.")

	def safe_set_joint_positions_velocities(self, positions, velocities):
		# The safety layer
		cur_safe = self.safe.test_new_configuration(positions)

		# Compute if the pose is safe
		pose_thresh = 0.25 # in rad
		cur_pose = self.get_positions(False)

		pose_dist = np.linalg.norm(positions - cur_pose)

		# Comput if the velocity is safe
		vel_thresh = 2.0
		cur_vel = self.get_velocities(False)

		vel_dist = np.linalg.norm(velocities - cur_vel)

		if pose_dist > pose_thresh:
			print("Next provided pose is too far. Aborting the current configuration")
		elif vel_dist > vel_thresh:
			print("Next provided velocity is too fast. Aborting the current configruation")
		elif cur_safe == False:
			print("Robot will hit the table!!! Aborting the current configuration.")
		else:
			self.set_joint_positions_velocities(positions, velocities, is_safe=True) # for impedance control
