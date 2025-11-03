import sys
import rospy
import numpy as np
from math import pi
from time import perf_counter

from core.interfaces import ArmController
from lib.potentialFieldPlanner import PotentialFieldPlanner 

from lib.loadmap import loadmap
from copy import deepcopy


starts = [np.array([0, -1, 0, -2, 0, 1.57, 0]),
          np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707])]
goals = [np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7]),
         np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])]
mapNames = ["map1",
            "map2",
            "map3",
            "map4",
            "emptyMap"]
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython potentialField_demo.py 1\n\tpython potentialField_demo.py 2\n\tpython potentialField_demo.py 3 ...")
        exit()

    rospy.init_node('potentialfield')

    arm = ArmController()
    index = int(sys.argv[1])-1
    print("Running test "+sys.argv[1])
    print("Moving to Start Position")
    arm.safe_move_to_position(starts[index])
    map_struct = loadmap("../../maps/"+mapNames[index] + ".txt")
    print("Map = "+ mapNames[index])

    print("Starting to plan")
    start = perf_counter()
    planner = PotentialFieldPlanner()
    path = planner.plan(deepcopy(map_struct), deepcopy(starts[index]), deepcopy(goals[index]))
    stop = perf_counter()
    dt = stop - start
    print("Potential Field took {time:2.2f} sec. Path is.".format(time=dt))
    print(path)
    input("Press Enter to Send Path to Arm")

    gap = 100
    for i in range(0, path.shape[0], gap):
        arm.safe_move_to_position(path[i, :])
    print("Trajectory Complete!")
