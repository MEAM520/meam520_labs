import sys
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs
import visualization_msgs
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity_null import IK_velocity_null
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
from lib.calcManipulability import calcManipulability

#####################
## Rotation Helper ##
#####################

def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


##################
## Follow class ##
##################

class JacobianDemo():
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """
    active = False # When to stop commanding arm
    start_time = 0 # start time
    dt = 0.03 # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher('/vis/trace', geometry_msgs.msg.PointStamped, queue_size=10)
    ellipsoid_pub = rospy.Publisher('/vis/ellip', visualization_msgs.msg.Marker, queue_size=10)
    counter = 0
    x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
    last_iteration_time = None

    visualize_mani_ellipsoid = False

    ##################
    ## TRAJECTORIES ##
    ##################

    def eight(t,fx=0.3,fy=0.6,rx=.1,ry=.1):
        """
        Calculate the position and velocity of the figure 8 trajector

        Inputs:
        t - time in sec since start
        fx - frequecny in rad/s of the x portion
        fy - frequency in rad/s of the y portion
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        # Lissajous Curve
        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
        xdes = x0 + np.array([rx*sin(fx*t),ry*sin(fy*t),0])
        vdes = np.array([rx*fx*cos(fx*t),ry*fy*cos(fy*t),0])

        Rdes = np.diag([1., -1., -1.])
        ang_vdes = 0.0 * np.array([1.0, 0.0, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    def ellipse(t,f=0.5,ry=.15,rz=.10):
        """
        Calculate the position and velocity of the figure ellipse trajector

        Inputs:
        t - time in sec since start
        f - frequecny in rad/s of the trajectory
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position

        ## STUDENT CODE GOES HERE

        # TODO: replace these!
        xdes = JacobianDemo.x0+np.array([0,ry-ry*cos(f*t), rz*sin(f*t)])
        vdes = np.array([0,ry*f*sin(f*t), rz*f*cos(f*t)])
        Rdes = np.diag([1., -1., -1.])
        ang_vdes = 0.0 * np.array([1.0, 0.0, 0.0])
        ## END STUDENT CODE
        
        return Rdes, ang_vdes, xdes, vdes

    def line(t,f=0.1,L=.15):
        """
        Calculate the position and velocity of the line trajector

        Inputs:
        t - time in sec since start
        f - frequecny in Hz of the line trajectory
        L - length of the line in meters

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """
        ## STUDENT CODE GOES HERE
        x0 = np.array([0.307,0,0.487]) #corresponds to neutral position
        # TODO: replace these!
        xdes = np.array([0.307,L-L*cos(f*t),0.487])
        vdes = np.array([0,L*f*sin(f*t),0])

        # Example for generating an orientation trajectory
        # The end effector will rotate around the x-axis during the line motion
        # following the changing ang
        ang = -np.pi + (np.pi/4.0) * sin(f*t)
        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)

        ang_v = (np.pi/4.0) * f * cos(f*t)
        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        ## END STUDENT CODE
        return Rdes, ang_vdes, xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'endeffector'
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)

    def show_manipulability_ellipsoid(self, M):
        eigenvalues, eigenvectors = np.linalg.eig(M)

        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = "endeffector"
        marker.header.stamp = rospy.Time.now()
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        #axes_len = np.sqrt(eigenvalues)

        marker.scale.x = eigenvalues[0]
        marker.scale.y = eigenvalues[1]
        marker.scale.z = eigenvalues[2]

        R = np.vstack((np.hstack((eigenvectors, np.zeros((3,1)))), \
                       np.array([0.0, 0.0, 0.0, 1.0])))
        q = quaternion_from_matrix(R)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        self.ellipsoid_pub.publish(marker)

    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:

            try:
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                Rdes, ang_vdes, xdes, vdes = trajectory(t)

                # get current end effector position
                q = state['position']

                joints, T0e = self.fk.forward(q)

                R = (T0e[:3,:3])
                x = (T0e[0:3,3])

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 5
                v = vdes + kp * (xdes - x)

                # Rotation
                kr = 5
                omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()


                ## STUDENT CODE MODIFY HERE, DEFINE SECONDARY TASK IN THE NULL SPACE
                lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
                upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
                q_e = lower + (upper - lower) / 2
                k0 = 1.0

                # Velocity Inverse Kinematics
                dq = IK_velocity_null(q,v, omega, - k0 * (q - q_e)).flatten()

                # Visualize Manipulability
                mu, M = calcManipulability(q)

                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()

                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()

                new_q = q + self.dt * dq

                arm.safe_set_joint_positions_velocities(new_q, dq)

                # Downsample visualization to reduce rendering overhead
                self.counter = self.counter + 1
                if self.counter == 10:
                    self.show_ee_position()
                    if self.visualize_mani_ellipsoid:
                        self.show_manipulability_ellipsoid(M)
                        print('Manipulability Index',mu)
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython follow.py line\n\tpython follow.py ellipse\n\tpython follow.py eight")
        exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    if sys.argv[1] == 'line':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.line)
    elif sys.argv[1] == 'ellipse':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.ellipse)
    elif sys.argv[1] == 'eight':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.eight)
    else:
        print("invalid option")
        exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # q = np.array([ 0,    0,     0, 0,     0, pi, 0.75344866 ])
    # arm.safe_move_to_position(q)
    # mu, M = calcManipulability(q)
    # print(mu)
    # JD.show_manipulability_ellipsoid(M)
    
    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")
