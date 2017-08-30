#
# this file was created by a computer. trust it.
#
XILINX_SDX ?= $(XILINX_SDACCEL)

# compiler tools
XILINX_VIVADO_HLS ?= $(XILINX_SDX)/Vivado_HLS

#SDX_CXX ?= $(XILINX_SDX)/bin/xcpp
SDX_CXX ?= gcc
XOCC ?= $(XILINX_SDX)/bin/xocc
EMCONFIGUTIL = $(XILINX_SDX)/bin/emconfigutil --od .
RM = rm -f
RMDIR = rm -rf

SDX_PLATFORM = xilinx:pea-c8k1-115:1ddr:4.0

# host compiler global settings
#CXXFLAGS += -DSDX_PLATFORM=$(SDX_PLATFORM) -D__USE_XOPEN2K8 -I/opt/Xilinx/SDx/2017.1/runtime/include/1_2/ -I/opt/Xilinx/SDx/2017.1/Vivado_HLS/include/ -O2 -Wall -c -fmessage-length=0 -std=c++14
CXXFLAGS += -DSDX_PLATFORM=$(SDX_PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_SDX)/runtime/include/1_2/ -I$(XILINX_SDX)/Vivado_HLS/include/ -O2 -Wall -c -fmessage-length=0 -std=c99
#LDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++ -L/opt/Xilinx/SDx/2017.1/runtime/lib/x86_64
LDFLAGS += -lxilinxopencl -lpthread -lm -L$(XILINX_SDX)/runtime/lib/x86_64

# kernel compiler global settings
XOCC_OPTS = -t hw --xdevice $(SDX_PLATFORM) --save-temps  --report system

#
# OpenCL kernel files
#

BINARY_CONTAINERS += lstm_container.xclbin

BUILD_SUBDIRS += lstm_container
BINARY_CONTAINER_1_OBJS += lstm_container/lstm_input.xo
ALL_KERNEL_OBJS += lstm_container/lstm_input.xo
BUILD_SUBDIRS += lstm_container
BINARY_CONTAINER_1_OBJS += lstm_container/lstm_layer.xo
ALL_KERNEL_OBJS += lstm_container/lstm_layer.xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

#
# host files
#

HOST_OBJECTS += src/device_info.o
HOST_OBJECTS += src/testlstm.o
HOST_OBJECTS += src/wtime.o

HOST_EXE = testlstm 

BUILD_SUBDIRS += src/

EMCONFIG_FILE = emconfig.json

#
# primary build targets
#

.PHONY: all clean
all: $(BINARY_CONTAINERS) $(HOST_EXE) $(EMCONFIG_FILE)

clean:
	-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES) $(EMCONFIG_FILE) $(HOST_EXE) $(HOST_OBJECTS)
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) _xocc*
	-$(RMDIR) .Xil

.PHONY: incremental
incremental: all

include makeemconfig.mk

#
# binary container: lstm_container.xclbin
#

lstm_container/lstm_input.xo: ../src/lstm.cl
	@mkdir -p $(@D)
	-@$(RM) $@
	$(XOCC) $(XOCC_OPTS) -c -k lstm_input --messageDb $(subst .xo,.mdb,$@) -I"$(<D)" --xp misc:solution_name=_xocc_compile_lstm_container_lstm_input -o"$@" "$<"
	-@$(RMDIR) .Xil

lstm_container/lstm_layer.xo: ../src/lstm.cl
	@mkdir -p $(@D)
	-@$(RM) $@
	$(XOCC) $(XOCC_OPTS) -c -k lstm_layer --messageDb $(subst .xo,.mdb,$@) -I"$(<D)" --xp misc:solution_name=_xocc_compile_lstm_container_lstm_layer -o"$@" "$<"
	-@$(RMDIR) .Xil

lstm_container.xclbin: $(BINARY_CONTAINER_1_OBJS)
	$(XOCC) $(XOCC_OPTS) -l --nk lstm_input:1  --nk lstm_layer:1 --messageDb $(subst .xclbin,.mdb,$@) --xp misc:solution_name=_xocc_link_lstm_container -o"$@" $(+)
	-@$(RMDIR) .Xil

#
# host rules
#

src/testlstm.o: ../src/testlstm.c ../src/kerneldefs.h ../src/err_code.h
	@mkdir -p $(@D)
	$(SDX_CXX) $(CXXFLAGS) -o "$@" "$<"

src/%.o: ../src/%.c
	@mkdir -p $(@D)
	$(SDX_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(SDX_CXX) -o "$@" $(+) $(LDFLAGS)

#
# emulation configuration file
#

$(EMCONFIG_FILE): makeemconfig.mk
	$(EMCONFIGUTIL) --xdevice $(SDX_PLATFORM) --nd $(NUMBER_OF_DEVICES)
	-@$(RMDIR) TempConfig .Xil

