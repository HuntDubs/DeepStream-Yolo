################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Edited by Marcos Luciano
# https://www.github.com/marcoslucianops
#This is Hunter making an edit
################################################################################

#If CUDA_VER is not set, then throw an error and abort mission
CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

#If OPENCV is not set in command line, set it equal to 0
OPENCV?=
ifeq ($(OPENCV),)
	OPENCV=0
endif

#Set variables
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

#If OPENCV is set to 1
ifeq ($(OPENCV), 1)
	COMMON= -DOPENCV
	CFLAGS+= $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
	LIBS+= $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

#Set external library flags to use while building the object files
LIBS+= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

# Set all header files
INCS:= $(wildcard *.h)

# Set source files to all .cpp files, but only include calibrator.cpp if we are using OPENCV
# We are not using OPENCV, so a calibrator.o object file is not created
SRCFILES:= $(filter-out calibrator.cpp, $(wildcard *.cpp))

ifeq ($(OPENCV), 1)
	SRCFILES+= calibrator.cpp
endif

#Include the files in the layers subfolder and the CUDA files in the source files
SRCFILES+= $(wildcard layers/*.cpp)
SRCFILES+= $(wildcard *.cu)

# We want to compile the lib into this shared object file
TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

# We want to compile the c++ files into object files
TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

# Compile all into the TARGET_LIB
all: $(TARGET_LIB)

# for every target X.o, if there exists a file named X.cpp, do the following
%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

# for every target X.o, if there exists a file named X.cu, do the following
# NVCC specifies the compiler for CUDA files
%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

# Compile all the object files into the main shared object file
$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

# Remove all the object files and the shared object file
clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
