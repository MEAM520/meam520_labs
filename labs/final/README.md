# MEAM 520 Final Project Simulation Environment (Dual Franka Setup)

## Overview

We have put significant effort into supporting **two Franka arms operating within the same ROS master**.

From lecture, you’ve seen how ROS enables communication across robotic systems. However, enabling **multiple robots in a shared environment** required extensive changes across several Franka ROS packages.

Because of this, the easiest way to distribute a stable setup is through a **new VM image**:

**Download here:** `<insert link>`

**Important:**  
Your previous VM image with the packages are **will NOT work** for two robots.  
You must:
- Use the new VM
- Manually transfer any previous lab code you want to reuse for the final project

---

## Launching the Simulation


```bash
cd ~/meam520_ws

roslaunch meam520_labs final.launch
```
This will launch both Gazebo and RViz. Focus on the Gazebo for your primary simulation environment.

RViz is meant to provide camera simulation but the dual-arm vision is not yet fully implemented. Vision/simulation teams are welcome to discuss and extend RViz functionality if desired.

---

## Simulation Environment

The setup consists of **two Franka arms** and a turntable positioned between the two robots.

<insert photo>

### Customizing the World
The environment is defined in the following file:
`~/meam520_ws/src/meam520_labs/ros/meam520_labs/worlds/final.world`

**Simulation team should take a look to figure out how to modify:**
* You should replace the objects to suit your project needs.
* The 4 cubes currently in the world are examples. You can use them as reference to initialize dispensers, cups, etc.

---

## Controlling the Robots

Control logic remains consistent with previous labs via functions available in `~/meam520_ws/src/meam520_labs/core/interfaces.py`. The key distinction for this setup is the **Robot ID**.

### Initializing Controllers
You must initialize two separate controller instances and specify the ID to differentiate between the arms.

```python
# Initialize the controllers
arm1 = ArmController(id=1)  # Left robot
arm2 = ArmController(id=2)  # Right robot
```

You then call commands on the appropriate arm.

To see a demo of controlling both arms:

```bash
cd ~/meam520_ws/src/meam520_labs/labs/final/

python final.py
```
You will notice that commands execute **sequentially**.

---

## GitHub Setup

To push your code to your own repository, you must add the destination of your repo:

```bash
cd ~/meam520_ws/src/meam520_labs

git remote add origin git@github.com:<your-username>/meam520_labs.git
```

You only need to do this once. After that, use your usual workflow:

```bash
git add .
git commit -m "your message"
git push origin main
```
## Notes and Support
This is an experimental setup and has not been fully tested.

If you run into issues, reach out to Amy Luo and Hojin Choi.
