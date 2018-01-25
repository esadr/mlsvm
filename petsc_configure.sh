export PETSC_DIR=$PWD
./configure PETSC_ARCH=linux-no-debug \
--with-cc=gcc \
--with-cxx=g++ \
--with-clanguage=c++ \
--with-gnu-compilers=1 \
--with-mpi-compilers=1 \
--with-debugging=0 \
--with-shared-libraries=1 \
--download-openmpi=1 \
--download-metis=1 \
--download-parmetis=1 \
--download-blacs=1 \
--download-cmake \
--download-f2cblaslapack=1 
