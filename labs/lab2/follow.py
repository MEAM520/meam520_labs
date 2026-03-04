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

from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK

######################
## Rotation Helpers ##
######################


def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 1x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    Rrel = R_curr.T @ R_des
    skew = 1 / 2 * (Rrel - Rrel.T)
    u = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    omega = R_curr @ u

    return omega


##################
## Follow class ##
##################


class JacobianDemo:
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """

    active = False  # When to stop commanding arm
    start_time = 0  # start time
    dt = 0.03  # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher(
        "/vis/trace", geometry_msgs.msg.PointStamped, queue_size=10
    )
    ellipsoid_pub = rospy.Publisher(
        "/vis/ellip", visualization_msgs.msg.Marker, queue_size=10
    )
    counter = 0
    x0 = np.array([0.307, 0, 0.487])  # corresponds to neutral position
    last_iteration_time = None

    ##################
    ## TRAJECTORIES ##
    ##################

    def line(t, f=1.0, L=0.35):
        """
        Calculate the position and velocity of the line trajector

        Inputs:
        t - time in sec since start
        f - frequency in rad/s of the line trajectory
        L - length of the line in meters

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        Rdes = 3x3 np array of target end effector orientation in the world frame
        ang_vdes = 0x3 np array of target end effector orientation velocity in the
        rotation vector representation in the world frame
        """
        xdes = JacobianDemo.x0 + np.array([0, L * sin(f * t), 0])
        vdes = np.array([0, L * f * np.cos(f * t), 0])

        # Example for generating an orientation trajectory
        # The end effector will rotate around the x-axis during the line motion
        # following the changing ang
        ang = -np.pi + (np.pi / 4.0) * sin(f * t)
        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)

        ang_v = (np.pi / 4.0) * f * cos(f * t)
        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    def eight(t, fx=0.5, fy=1.0, rx=0.15, ry=0.1):
        """
        Calculate the position and velocity of the figure 8 trajector

        Inputs:
        t - time in sec since start
        fx - frequency in rad/s of the x portion
        fy - frequency in rad/s of the y portion
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        Rdes = 3x3 np array of target end effector orientation in the world frame
        ang_vdes = 0x3 np array of target end effector orientation velocity in the rotation vector representation in the world frame
        """
        # TODO: Implement linear and orientation components of this trajectory
        # Test extensively in simulation before your robot lab!

        ## END STUDENT CODE
        return Rdes, ang_vdes, xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "endeffector"
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)

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
                q = state["position"]

                joints, T0e = self.fk.forward(q)

                R = T0e[:3, :3]
                x = T0e[0:3, 3]
                curr_x = np.copy(x.flatten())

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 0.01
                v = vdes + kp * (xdes - curr_x)

                # Rotation
                kr = 0.01
                omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

                # Velocity Inverse Kinematics
                dq = IK_velocity(q, v, omega).flatten()

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
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage:\n\tpython follow.py line\n\tpython follow.py eight")
        exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    if sys.argv[1] == "line":
        callback = lambda state: JD.follow_trajectory(state, JacobianDemo.line)
    elif sys.argv[1] == "eight":
        callback = lambda state: JD.follow_trajectory(state, JacobianDemo.eight)
    else:
        print("invalid option")
        exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # q = np.array([ 0,    0,     0, 0,     0, pi, 0.75344866 ])
    # arm.safe_move_to_position(q)

    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")
