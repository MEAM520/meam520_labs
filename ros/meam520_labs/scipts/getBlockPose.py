#!/usr/bin/env python3

from operator import pos
import rospy
import tf
from tf.transformations import euler_from_quaternion

from apriltag_ros.msg import AprilTagDetectionArray
from meam520_labs.msg import BlockDetection, BlockDetectionArray
from geometry_msgs.msg import PoseStamped

import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


H_0_1 = np.array([[-1,0,0,0], [0,0,-1,0], [0,-1,0,0], [0,0,0,1]])
H_0_2 = np.array([[0,1,0,0], [0,0,-1,0], [-1,0,0,0], [0,0,0,1]])
H_0_3 = np.array([[0,-1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
H_0_4 = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]])
H_0_5 = np.array([[-1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]])


def get_populated_msg(block_id, block_dict, trans_rot):
		block_detection = BlockDetection()

		block_detection.pose.header.stamp = rospy.Time.now()
		block_detection.pose.header.frame_id = 'camera'
		rot_quat = tf.transformations.quaternion_from_euler(trans_rot[1][0], trans_rot[1][1], trans_rot[1][2])
		block_detection.id = block_id
		block_detection.dynamic = block_dict[block_id][3]
		block_detection.pose.pose.position.x = trans_rot[0][0]
		block_detection.pose.pose.position.y = trans_rot[0][1]
		block_detection.pose.pose.position.z = trans_rot[0][2]
		
		block_detection.pose.pose.orientation.x = rot_quat[0]
		block_detection.pose.pose.orientation.y = rot_quat[1]
		block_detection.pose.pose.orientation.z = rot_quat[2]
		block_detection.pose.pose.orientation.w = rot_quat[3]

		return block_detection

def callback(data):
	block_detection_array = BlockDetectionArray()
	block_dict = defaultdict(lambda: np.array([[0,0,0],[0,0,0],0,0]))
	for i in range(len(data.detections)):
		tag_id = data.detections[i].id[0]
		if tag_id >= 300:
			pose_stamped = PoseStamped()
			pose_stamped.header = data.header
			pose_info = data.detections[i].pose.pose.pose
			pose_stamped.pose = pose_info
			center_tag_pub.publish(pose_stamped)
			continue
		
		br = tf.TransformBroadcaster()
		## br.sendTranform (translation, rotation, time, child, parent)
		br.sendTransform((0, 0, -0.025),
						[0, 0, 0, 1],
						rospy.Time.now(),
						"tag_{}_center".format(tag_id),
						"tag_{}".format(tag_id))
		try:
			# child, parent
			(trans,rot_quat) = listener.lookupTransform('camera', 'tag_{}_center'.format(tag_id), rospy.Time(0))
			## rotate the axis to align with tag_0 (and multiples)
			H_c_center = R.from_quat(rot_quat).as_matrix()
			H_c_center = np.append(H_c_center, np.array(trans).reshape(3,1), axis=1)
			H_c_center = np.append(H_c_center, np.array([0,0,0,1]).reshape(1,4), axis=0)

			if tag_id % 6 == 0:
				H_c_center_new = H_c_center
			elif tag_id % 6 == 1:
				H_c_center_new = H_c_center @ H_0_1
			elif tag_id % 6 == 2:
				H_c_center_new = H_c_center @ H_0_2
			elif tag_id % 6 == 3:
				H_c_center_new = H_c_center @ H_0_3
			elif tag_id % 6 == 4:
				H_c_center_new = H_c_center @ H_0_4
			elif tag_id % 6 == 5:
				H_c_center_new = H_c_center @ H_0_5
			
			trans = [H_c_center_new[0,3], H_c_center_new[1,3], H_c_center_new[2,3]]
			rot_quat = R.from_matrix(H_c_center_new[:3, :3]).as_quat()

			## can publish this transform for debugginf purpose
			# br.sendTransform(trans,
			# 			rot_quat,
			# 			rospy.Time.now(),
			# 			"tag_{}_center_new".format(tag_id),
			# 			"camera".format(tag_id))
						
			rot = euler_from_quaternion([rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]], axes='sxyz')
			block_dict[tag_id//6][0] += np.array(trans) 
			block_dict[tag_id//6][1] += np.array(rot) 
			block_dict[tag_id//6][2] += 1 
			block_dict[tag_id//6][3] = tag_id > 47 
		except Exception as e:
			print(e)

	for block_id in block_dict:
		trans_rot = block_dict[block_id][:2] / block_dict[block_id][2]
		block_detection = get_populated_msg(block_id, block_dict, trans_rot)
		block_detection_array.detections.append(block_detection)

		## can publish this transform for debugginf purpose
		# br.sendTransform((trans_rot[0][0], trans_rot[0][1], trans_rot[0][2]),
		# 				rot_quat,
		# 				rospy.Time.now(),
		# 				"block_{}_center".format(block_id),
		# 				"camera")

	block_pub.publish(block_detection_array)

	
if __name__ == '__main__':
	rospy.init_node('vision_pipeline_node', anonymous=False)
	listener = tf.TransformListener()
	rospy.Subscriber("tag_detections", AprilTagDetectionArray, callback)
	block_pub = rospy.Publisher('block_detections', BlockDetectionArray, queue_size=1)
	center_tag_pub = rospy.Publisher('center_tag_detection', PoseStamped, queue_size=1)
	rospy.spin()