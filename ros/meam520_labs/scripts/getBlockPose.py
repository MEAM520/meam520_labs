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

L = 0.025
Z_B = np.array([0.0, 0.0, 1.0])
# face 0 is the identity
H_0 = {}
H_0['cube'] = [np.eye(4), H_0_1, H_0_2, H_0_3, H_0_4, H_0_5]
H_0['rect'] = H_0['cube']   # same tag rotations, only offsets differ

def make_rect_offsets(H_0_cube, L):
    """Tag-frame center offsets for the rectangle block.
    Long axis is the block-frame z axis (face 0's tag normal)."""
    offsets, caps = [], []
    for k, H in enumerate(H_0_cube):
        R = H[:3, :3]                       # block frame -> tag k frame
        n_B = R.T @ np.array([0.0, 0.0, 1.0])   # tag k outward normal, block frame
        if abs(n_B @ Z_B) > 0.99:           # end cap
            d_b = -2 * L * n_B
            caps.append(k)
        else:                               # ring face
            d_b = -L * (n_B + Z_B)
        d_t = R @ d_b
        assert d_t[2] < 0, "face {}: center is in front of the tag".format(k)
        offsets.append(tuple(d_t))
    assert len(caps) == 2 and 0 in caps, "expected face 0 plus one opposite cap, got {}".format(caps)
    return offsets

CENTER_OFFSET = {
    'cube': [(0, 0, -0.025)] * 6,
    'rect': make_rect_offsets(H_0['cube'], L),
}

CENTER_TAG_RANGE = range(300, 400)
# Side-tag ID ranges, [start, end)
BLOCK_TYPES = {
    'cube': dict(
        start=0,
        end=120,              # 20 blocks * 6 faces -> IDs 0-119
        tags_per_block=6,
        static_from=48,       # IDs >= 48 are static, first 8 blocks are dynamic 
    ),
    'rect': dict(
        start=400,
        end=471,              # 12 blocks
        tags_per_block=6,     
        static_from=436,      # first 6 blocks are dynamic
    ),
}

def classify_tag(tag_id):
    """
    Map a side-tag ID to (block_type, block_index, face, is_static).
    Returns None if the ID is not a known block tag.
    """
    for name, cfg in BLOCK_TYPES.items():
        if cfg['start'] <= tag_id < cfg['end']:
            offset = tag_id - cfg['start']
            n = cfg['tags_per_block']
            is_static = (cfg['static_from'] is not None
                         and tag_id >= cfg['static_from'])
            return name, offset // n, offset % n, is_static
    return None

def get_populated_msg(block_key, block_dict, trans_rot):
                block_type, block_id = block_key
                block_detection = BlockDetection()

                block_detection.pose.header.stamp = rospy.Time.now()
                block_detection.pose.header.frame_id = 'camera'
                rot_quat = tf.transformations.quaternion_from_euler(trans_rot[1][0], trans_rot[1][1], trans_rot[1][2])
                block_detection.id = block_id
                block_detection.type = block_type
                block_detection.dynamic = block_dict[block_key][3]
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
		if tag_id in CENTER_TAG_RANGE:
			pose_stamped = PoseStamped()
			pose_stamped.header = data.header
			pose_info = data.detections[i].pose.pose.pose
			pose_stamped.pose = pose_info
			center_tag_pub.publish(pose_stamped)
			continue

                tag_info = classify_tag(tag_id)
                if info is None:
                        continue
                block_type, block_index, face, is_static = tag_info
        
		br = tf.TransformBroadcaster()
                br.sendTransform(CENTER_OFFSET[block_type][face],
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

                        H_c_center_new = H_c_center @ H_0[block_type][face]
                        
			# if tag_id % 6 == 0:
			# 	H_c_center_new = H_c_center
			# elif tag_id % 6 == 1:
			# 	H_c_center_new = H_c_center @ H_0_1
			# elif tag_id % 6 == 2:
			# 	H_c_center_new = H_c_center @ H_0_2
			# elif tag_id % 6 == 3:
			# 	H_c_center_new = H_c_center @ H_0_3
			# elif tag_id % 6 == 4:
			# 	H_c_center_new = H_c_center @ H_0_4
			# elif tag_id % 6 == 5:
			# 	H_c_center_new = H_c_center @ H_0_5
			
			trans = [H_c_center_new[0,3], H_c_center_new[1,3], H_c_center_new[2,3]]
			rot_quat = R.from_matrix(H_c_center_new[:3, :3]).as_quat()

			## can publish this transform for debugginf purpose
			# br.sendTransform(trans,
			# 			rot_quat,
			# 			rospy.Time.now(),
			# 			"tag_{}_center_new".format(tag_id),
			# 			"camera".format(tag_id))
						
			rot = euler_from_quaternion([rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]], axes='sxyz')
			# block_dict[tag_id//6][0] += np.array(trans)  # position xyz
			# block_dict[tag_id//6][1] += np.array(rot)    # rotation euler angles
			# block_dict[tag_id//6][2] += 1                
			# block_dict[tag_id//6][3] = tag_id > 47       # True if static, false if dynamic
                        key = (block_type, block_index)
                        block_dict[key][0] += np.array(trans)
                        block_dict[key][1] += np.array(rot)
                        block_dict[key][2] += 1
                        block_dict[key][3] = is_static
                        
		except Exception as e:
			print(e)

	for block_key in block_dict:
		trans_rot = block_dict[block_key][:2] / block_dict[block_key][2]
		block_detection = get_populated_msg(block_key, block_dict, trans_rot)
		block_detection_array.detections.append(block_detection)

		## can publish this transform for debugginf purpose
		# br.sendTransform((trans_rot[0][0], trans_rot[0][1], trans_rot[0][2]),
		# 				rot_quat,
		# 				rospy.Time.now(),
		# 				"block_{}_center".format(block_key),
		# 				"camera")

	block_pub.publish(block_detection_array)

	
if __name__ == '__main__':
	rospy.init_node('vision_pipeline_node', anonymous=False)
	listener = tf.TransformListener()
	rospy.Subscriber("tag_detections", AprilTagDetectionArray, callback)
	block_pub = rospy.Publisher('block_detections', BlockDetectionArray, queue_size=1)
	center_tag_pub = rospy.Publisher('center_tag_detection', PoseStamped, queue_size=1)
	rospy.spin()
