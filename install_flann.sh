#!/bin/bash

echo "Installing FLANN"

pushd ./flann

mkdir build
cd build/
sudo cmake ..
sudo make
sudo make install 

echo "FLANN is installed successfully!"
popd