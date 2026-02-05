import sys
from math import pi
import numpy as np

import rospy
import roslib
import tf
import geometry_msgs.msg


from core.interfaces import ArmController

from lib.calculateFK import FK

rospy.init_node("visualizer")

# Using your solution code
fk = FK()

#########################
##  RViz Communication ##
#########################

tf_broad  = tf.TransformBroadcaster()
point_pubs = [
    rospy.Publisher('/vis/joint'+str(i), geometry_msgs.msg.PointStamped, queue_size=10)
    for i in range(7)
]

# Publishes the position of a given joint on the corresponding topic
def show_joint_position(joints,i):
    msg = geometry_msgs.msg.PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'world'
    msg.point.x = joints[i,0]
    msg.point.y = joints[i,1]
    msg.point.z = joints[i,2]
    point_pubs[i].publish(msg)

# Broadcasts a T0e as the transform from given frame to world frame
def show_pose(T0e,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(T0e),
        tf.transformations.quaternion_from_matrix(T0e),
        rospy.Time.now(),
        frame,
        "world"
    )

# Uses the above methods to visualize the full results of your FK
def show_all_FK(state):
    q = state['position']
    joints, T0e = fk.forward(q)
    show_pose(T0e,"endeffector")
    for i in range(7):
        show_joint_position(joints,i)


########################
##  FK Configurations ##
########################

# TODO: Try testing other configurations!

# The first configuration below matches the dimensional drawing in the handout
configurations = [
    np.array([ 0,    0,     0, -pi/2,     0, pi/2, pi/4 ]),
    np.array([ pi/2, 0,  pi/4, -pi/2, -pi/2, pi/2,    0 ]),
    np.array([ 0,    0, -pi/2, -pi/4,  pi/2, pi,   pi/4 ]),
]

####################
## Test Execution ##
####################

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("usage:\n\tpython visualize.py FK\n\tpython visualize.py IK")
        exit()

    arm = ArmController(on_state_callback=show_all_FK)

    if sys.argv[1] == 'FK':

        # Iterates through the given configurations, visualizing your FK solution
        # Try editing the configurations list above to do more testing!
        for i, q in enumerate(configurations):
            print("Moving to configuration " + str(i) + "...")
            arm.safe_move_to_position(q)
            if i < len(configurations) - 1:
                input("Press Enter to move to next configuration...")

        arm.safe_move_to_position(q)

    else:
        print("invalid option")
