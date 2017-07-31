XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:pea-c8k1-115:1ddr:4.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CPP := g++

OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

#CFLAGS += -std=c99
#CXXFLAGS := -Wall -Werror
#CLFLAGS := -g --xdevice $(DSA)
CLFLAGS := -j 12 --xdevice $(DSA)

#CLFLAGS += -t sw_emu
#CLFLAGS += -t hw_emu
#CLFLAGS += -t hw

HEADERS = device_picker.h err_code.h
CSRCS = device_info.c wtime.c testlstm.c

.PHONY: all
all: exe xclbin

.PHONY: xclbin
xclbin: kernel.xclbin

.PHONY: exe
exe: testlstm

testlstm: $(CSRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -o $@ $(CSRCS)
#	$(CC) $(CFLAGS) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -o $@ $(CSRCS)

kernel.xclbin: lstm.cl
	$(XOCC) $(CLFLAGS) $< -o $@

clean:
	rm -rf kernel.xclbin testlstm xocc* sdaccel*
