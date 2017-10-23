#!/usr/bin/env bash
set -e
git config --global user.name reinvantveer
git config --global user.email 'rein.van.t.veer@geodan.nl'

cd ~
sudo apt-get update
sudo apt-get install -y libgeos-dev python3-tk  # reinstall python3?
sudo pip3 install --upgrade keras  # check ~/.keras/keras.json for correct settings!
sudo pip3 install shapely
# Install magenta requirement cuda 8.0 v6 for tf 1.2
# From https://gitlab.com/nvidia/cuda/blob/c5e8c8d7a9fd444c4e45573f36cbeb8f4e10f71c/8.0/runtime/cudnn6/Dockerfile
# And https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar xvzf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig

sudo apt-get install -y lxde
# sudo rm /home/ubuntu/.Xauthority
sudo startlxde
sudo add-apt-repository -y ppa:x2go/stable
sudo apt-get update
sudo apt-get install -y x2goserver x2goserver-xsession
wget https://download.jetbrains.com/python/pycharm-community-2017.2.3.tar.gz
tar xvzf pycharm-community-2017.2.3.tar.gz

sudo apt-get install numlockx
sudo sed -i 's|^exit 0.*$|# Numlock enable\n[ -x /usr/bin/numlockx ] \&\& numlockx on\n\nexit 0|' /etc/rc.local
echo "/usr/bin/numlockx on" | sudo tee -a /etc/X11/xinit/xinitrc
sudo reboot

# Install Jenkins
wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -
echo "deb https://pkg.jenkins.io/debian-stable binary/" | sudo tee /etc/apt/sources.list.d/jenkins-debian-stable.list
sudo apt-get update
sudo apt-get install jenkins