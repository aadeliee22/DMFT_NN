#!/usr/bin/env bash
INSTALL_PREFIX_TRIQS=`pwd`/triqs.build
NCPU=20
#COMPILER_FLAGS="-DCMAKE_CXX_COMPILER=`which g++` -DCMAKE_C_COMPILER=`which gcc`"

if ! [ -e triqs.src ]; then
	git clone https://github.com/TRIQS/triqs triqs.src
fi

if ! [ -e nrgljubljana_interface.src ]; then
	git clone https://github.com/TRIQS/nrgljubljana_interface nrgljubljana_interface.src
fi


mkdir -p triqs.build
cd triqs.build
eval "cmake ../triqs.src $COMPILER_FLAGS -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX_TRIQS"

make -j $NCPU

#make test
make install

cd ../


mkdir -p nrgljubljana_interface.build
cd nrgljubljana_interface.build

source $INSTALL_PREFIX_TRIQS/triqsvars.sh

eval "cmake ../nrgljubljana_interface.src $COMPILER_FLAGS"

make -j $NCPU 
make test
make install
