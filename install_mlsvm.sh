#!/bin/bash

#check for cmake installation



cp petsc_configure.sh ./petsc/
pushd ./petsc

./petsc_configure.sh

#call install_flann.sh


#define the PETSC_DIR, PETSC_ARCH
