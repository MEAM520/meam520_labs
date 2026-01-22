# MEAM5200 Frequently Asked Questions
---
# Lab 0

### Physical vs. virtual machine

You will need to set up the correct environment in Ubuntu 20.04, install ROS Noetic, and include the dependencies for the lab (e.g. panda_simulator). If you don't have prior experience with these, I would recommend using the virtual machine instead. It will save you a lot of time.

If you have prior experience with ROS and native Ubuntu, feel free to directly follow the instructions [here](https://github.com/MEAM520/meam520_labs/blob/main/README.md)

### Ubuntu Version: 20.04

For this class, the simulator must be run on Ubuntu 20.04 with ROS noetic installed. You can follow the standard installation instructions for [Ubuntu 20.04](https://phoenixnap.com/kb/install-ubuntu-20-04) and [ROS noetic](http://wiki.ros.org/noetic/Installation).

There is no official ROS release for 21.04, so it might cause problems.

As for Ubuntu 18.04, it supports the older version of ROS (Melodic), which could also cause problems due to the version mismatch.

Therefore, we recommend using the Virtual Machine Ubuntu 20.04 or switching to native Ubuntu 20.04 instead. 

### Network Connection Issues

For some different versions of the host machines, you might need to check and change your network settings.

- "Fail to Fetch... " during sudo apt install open-vm-tools-desktop

Shut down the VM completely (not just suspend), then in VMware, go to the Virtual Machine Library, right-click on MEAM520F22, and choose Settings. Under Network Adapter, make sure the connection is set to "Share with my Mac" or "NAT" if you are using Windows or Ubuntu 22.04 or newer. Relaunch the VM, then go to the top menu bar, select Virtual Machine - Network Adapter, and confirm that NAT is selected.


### Error with optimizing VMWare Player 17.0

It's come to our attention that some of you are having issues of accessing VM settings on VMware Workstation Player 17. You might follow the instructions below to fix this issue.

- Solution 1: Modifying the configuration file directly.

1. Make sure you close the VMware app, then open the folder where the MEAM5200 virtual machine is located.
2. In that folder, For Linux, locate one file with the extension .vmx; For Windows, locate one file with type "VMware virtual machine configuration"
3. Open it with any text editor or IDE, for example, NotePad on Windows.
4. If the value for virtualhw.version is not "19" , please change it to "19" and save the file.
5. Reopen VMware software, try to edit settings to see if it works.
If your problem is solved, you don't need to modify the configuration file anymore. Otherwise, you can edit number of cores and storage size directly in the vmx file: [ref](https://kb.vmware.com/s/article/205n).

- Solution 2: You may want to install VMWare Player 16.0 instead.

### Incorrect username and password for logging into Virtual PC Lab

You should use your PennKey username and password. If it does not work, you need to change your password and try again. 

- You can use the very same password!!! For some reason the password changing process itself will fix this! 

### Atom Editor does not Save Python code

- Solution 1: 
1. Inside your virtual machine, open up the terminal (make sure you close Atom)
2. Enter this command `cd ~/.local/share`
3. Then enter the following command `rm -rf keyrings`
4. Recreate `keyrings` folder by typing this command `mkdir keyrings`
5. Open up Atom and now you'll see a prompt asking for password, simply type any password and press enter
6. Now you should be able to edit and save files.
   
- Solution 2: Try using some other editor like vscode. You can always use 'nano' in the terminal if nothing else works.

### Script not appearing in ATOM

The path you typed in the terminal is not correct. (This might be due to directly copy/pasting from handout. The ~ sign is a little bit different when you paste it, delete the ~ in the terminal and type it yourself.)

### No module named 'core'

Please make sure the folder name is **meam520_labs** exactly, not "meam5200_labs" or "meam5200-labs"

### Why does my computer blue screen/crash/VMware Workstation Unrecoverable Error: (vcpu-xx) when I open the virtual environment?

1. Enabled Virtual Technol（intel VT-x/EPT or AMD-V/RVI，this option can be turned on somewhere in the Bios. After doing that, you can see 'Virtualization Enabled' in the Task manager as Fig below.
2. Go to Windows Feature, and enable 'Virtual Machine Platform'.

### I have assigned many compute cores to the VM, why is it still running slow?

This happens on CPUs with Intel's 12th and 13th generation architectures (Alder Lake Soc & Raptor Lake Soc).  INTEL divides the cores into E-cores (Efficiency) and P-cores (Performance), and they are meant to boost performance and reduce energy consumption. This may be great in most situation, but if we do not change the default settings, the VMware will run on E-cores, which will slow down the program. As you can see in the Fig below, the CPU usage is very low.

Solution: Right-click on VMware Workstation and "run as administor", and then the CPU usage comes to normal.

### I compiled the program correctly, but I can't see the robotic arm?

Go to Virtual Machine Settings>Hardware>Display, and disable the 'Accelerate 3D graphics' in VMware.

### roslaunch error, RLException: unable to contact my own server

1. Use the lines below on your command window

`sudo apt install net-tools`

`ifconfig`

on the second paragraph second line it will give you an ip (inet: IP ADDRESS)

2. Open ".bashrc" file (In your command window, type: `gedit ~/.bashrc` 
3. In .bashrc file add the following lines:
   
`export ROS_MASTER_URI=http://IP_OF_LAPTOP:11311`

`export ROS_HOSTNAME=IP_OF_LAPTOP`

- Replace the IP_OF_LAPTOP with your own IP address

### `AttributeError: module 'numpy' has no attribute 'float'`

np.float is deprecated in numpy newer than 1.24. The default numpy version for the virtual machine environment is 1.17, so our code still use np.float. The reason your are getting this error is that you are using numpy >= 1.24 in which np.float does not exist.

Solution: downgrade numpy to versions <1.24


### Fail to revert to working Ubuntu image for Mac host

If it does not work, you can [take a new snapshot](https://techdocs.broadcom.com/us/en/vmware-cis/desktop-hypervisors/fusion-pro/13-0/using-vmware-fusion/protecting-your-virtual-machines/snapshots/take-a-snapshot.html) for your future debugging. 


