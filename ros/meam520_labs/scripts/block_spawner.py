#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty, EmptyRequest
from math import sin, cos, pi
import numpy as np
from time import sleep
import os
import subprocess

from std_msgs.msg import Float64

import tf

rospy.init_node('block_spawner')

from gazebo_msgs.srv import SpawnModel
spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)


path = os.path.expanduser('~/meam520_ws/src/meam520_labs/ros/meam520_labs/urdf/cube.xacro')
path2 = os.path.expanduser('~/meam520_ws/src/meam520_labs/ros/meam520_labs/urdf/double_cube.xacro')
command = 'rosrun xacro xacro '+path+' color:='
command2 = 'rosrun xacro xacro '+path2+' color:='
colors = ["'.8 .8 .8 1'","'.8 .8 .1 1'"]
xmls = [subprocess.check_output(command+color,shell=True).decode('utf-8') for color in colors]
xmls2 = [subprocess.check_output(command2+color,shell=True).decode('utf-8') for color in colors]
team = rospy.get_param("team") # 'red' or 'blue'


count = 0
def place(x,y,z,place_type,block_type):
    global count
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    r = np.random.randint(0,3) * 2 * np.pi / 4
    if block_type=='cube':
        p = np.random.randint(0,3) * 2 * np.pi / 4
    elif block_type=='rect':
        p = 0.0
    y = np.random.rand(1) * 2 * pi
    q = tf.transformations.quaternion_from_euler(r,p,y)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    count = count + 1
    success = False
    while not success and not rospy.is_shutdown():
        try:
            if block_type=='cube':
                xml_str = xmls
            elif block_type=='rect':
                xml_str = xmls2
            success = spawn_model_client(
                    model_name=block_type+str(count)+'_'+place_type,
                    model_xml= xml_str[0] if place_type=='static' else xml_str[1],
                    robot_namespace='/foo',
                    initial_pose=pose,
                    reference_frame='world')
        except:
            print('Waiting to spawn cubes...')
            sleep(1)


def noise(radius):
    return radius * (np.random.rand(1) - .5)

#options are 'cube' or 'rect' (new)
static_poses = np.array([[-1,-1], [1,-1], [-1,1], [1,1]])
n_base_rects = 1
base_static = static_poses[np.random.choice(4,size=3,replace=False),:]
base_cubes = base_static[:-n_base_rects,:]
base_rect = base_static[-n_base_rects:,:]

types = ['cube', 'rect']
base_poses = [base_cubes, base_rect]
rect_xyz = []
for type_name, pose_list in zip(types, base_poses):
    for i,j in pose_list:
        x = .562 + 2.5*.0254 * i
        y = 1.159 + 2.5*.0254 * j
        y = -y if team=='red' else y
        z = .23
        place(x + noise(.025) ,y + noise(.025),z,'static',type_name)
        if type_name == 'rect':
            rect_xyz.append([x,y,z])
for x,y,z in rect_xyz:
    z += .052
    place(x + noise(.01),y + noise(.01), z,'static','cube')
        

# n = 1
n = 8 # ideally an even number

# dynamic rectangles are opposite each other
dyn_rect = [np.random.choice(n)]
dyn_rect.append((dyn_rect[0] + int(n/2))%8)
dyn_cube = np.delete(np.arange(n), dyn_rect)

r = 9.5*.0254
dyn_poses = []
for i in range(n):
    if i in dyn_rect:
        type_name = 'rect'
        ri = r - .0508
    elif i in dyn_cube:
        type_name = 'cube'
        ri = r
    else:
        print('problem: dynamic index', i)
    place((ri + noise(.0254)) * cos(2*pi/n * (i + noise(pi/n))),\
          (ri + noise(ri/5)) * sin(2*pi/n * (i + noise(pi/n))),\
          .23,
          'dynamic', type_name)



spin_pub = rospy.Publisher('/turntable/turntable/turntable_controller/command',Float64,queue_size=1)
msg = Float64()
msg.data = 1000 # sending large rpm value once

rospy.loginfo('spinning turntable...')
r = rospy.Rate(1) # Hz
while not rospy.is_shutdown():
    spin_pub.publish(msg)
    msg.data = .0523 # about .5 rpm
    r.sleep()
