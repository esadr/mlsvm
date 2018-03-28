#!/bin/bash

echo "Installing FLANN"

pushd ./flann

sudo apt-get install cmake -y

mkdir build
cd build/
sudo cmake ..
sudo make
sudo make install 

echo "FLANN is installed successfully!"
popd