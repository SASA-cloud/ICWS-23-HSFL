# create the Virtual environment using Virtualenv
# downloading virtualenv
sudo apt-get install python3-pip python3-dev
sudo -H pip3 install virtualenv virtualenvwrapper -i https://pypi.tuna.tsinghua.edu.cn/simple/
mkdir $HOME/.virtualenvs

# some configurations
vim ~/.bashrc
# adding these things on the ~./bashrc
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
source ~/.bashrc

# create a virtual environment "pytorch" with python3.6
mkvirtualenv -p python3 pytorch


# downloading the torch wheel (torch1.8.0) from: https://www.notion.so/pytorch-8f6a315fa06e44e69a86a8f32b2f100c?pvs=4#5e20d17414b34eb2b126a7c036edebbe
# we download the torch wheel on the laptop and then transfer it to the edge devices:
scp -r ./torch1.8-torchvision0.9.0  username@ip: download_destination_location

# installing the torch wheel
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# configuration on the system
sudo vim ~/.bashrc
# adding these things on the ~./bashrc
    export OPENBLAS_CORETYPE=ARMV8 # 写到最后作为环境变量
source ~/.bashrc
workon pytorch # this is the virtual environment created using Virtualenv

# downloading and installing the torchvision wheel
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
$ cd torchvision
$ export BUILD_VERSION=0.9.0  # where 0.x.0 is the torchvision version  
$ python3 setup.py install --user
$ cd ../  # attempting to load torchvision from build dir will result in import error
$ pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6

# Outputs:
# Using /home/nisl/.virtualenvs/pytorch/lib/python3.6/site-packages
# Finished processing dependencies for torchvision==0.9.0a0+01dfa8e

# adding the package into the virtual environment "pytorch"
add2virtualenv /home/nisl/.local/lib/python3.6/site-packages/torchvision-0.9.0a0+01dfa8e-py3.6-linux-aarch64.egg
add2virtualenv /home/nisl/.local/lib/python3.6/site-packages/torchvision-0.9.0-py3.6-linux-aarch64.egg
pip list

# Outputs:
# Package           Version
# ----------------- ---------------
# Cython            0.29.33
# dataclasses       0.8
# numpy             1.19.5
# pip               21.3.1
# setuptools        59.6.0
# torch             1.8.0
# torchvision       0.9.0a0+01dfa8e
# typing_extensions 4.1.1
# wheel             0.37.

