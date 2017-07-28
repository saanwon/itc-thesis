#CPP := g++
#IOC := ioc64

CFLAGS += -I/usr/local/include -L/usr/local/lib -I/usr/local/share/pocl/include -rdynamic -lOpenCL
#CXXFLAGS += -std=gnu++0x
#CLFLAGS +=

HEADERS = device_picker.h err_code.h
CSRCS = device_info.c wtime.c testlstm.c
CLSRCS = lstm.cl

.PHONY: all
#all: exe xclbin
all: exe

#.PHONY: xclbin
#xclbin: kernel.xclbin

.PHONY: exe
exe: testlstm

testlstm: $(CSRCS) $(HEADERS)
	gcc -g -o $@ $(CSRCS) $(CFLAGS)

#kernel.xclbin: $(CLSRCS)
#	$(IOC) $(CLFLAGS) -cmd=build -input=$(CLSRCS) -ir=$@

clean:
	rm -rf kernel.xclbin testlstm
