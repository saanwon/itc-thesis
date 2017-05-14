# Define the solution for SDAccel
create_solution -name gru -dir . -force
add_device "xilinx:pea-c8k1-115:1ddr:4.0"

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE"  -objects [current_solution]

# Host Source Files
add_files testgru.c device_info.c wtime.c

# Kernel Definition
create_kernel gru -type clc
add_files -kernel [get_kernels gru] "gru.cl"

# Define Binary Containers
create_opencl_binary cu_gru
set_property region "OCL_REGION_0" [get_opencl_binary cu_gru]
create_compute_unit -opencl_binary [get_opencl_binary cu_gru] -kernel [get_kernels gru] -name k1

# Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary cu_gru]
# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "cu_gru.xclbin"

# Compile the application to run on the accelerator card
#build_system

# Package the application binaries
#package_system

# run_system -args <command line arguments>
