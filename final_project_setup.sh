sudo mv /etc/apt/sources.list.d/ros-latest.list   /etc/apt/sources.list.d/ros-latest.list.disabled
sudo apt install -y gnupg

gpg --keyserver hkp://keyserver.ubuntu.com:80 \
    --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

gpg --export 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA | \
    sudo tee /usr/share/keyrings/ros1-snapshots-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/ros1-snapshots-archive-keyring.gpg] http://snapshots.ros.org/noetic/final/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros1-snapshots.list
sudo rm -f /var/lib/apt/lists/*packages.ros.org_ros_ubuntu*
sudo apt update
sudo apt install ros-noetic-velocity-controllers
sudo apt install ros-noetic-ros-control
