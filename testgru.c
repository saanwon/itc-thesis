#include <stdio.h>
#include <CL/opencl.h>

#include "err_code.h"
extern int output_device_info(cl_device_id device_id);
extern double wtime();       // returns time since some fixed past point (wtime.c)

static int
load_file_to_memory(const char *filename, char **result)
{
	size_t size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
		*result = NULL;
		return -1; // -1 means file opening fail 
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) {
		free(*result);
		return -2; // -2 means file reading fail
	}
	fclose(f);
	(*result)[size] = 0;
	return size;
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stderr, "Usage : %s <input file>\n", argv[0]);
		return EXIT_FAILURE;
	}

    cl_int          err;
	cl_device_id    device_id = NULL;		// compute device id

    // Find number of platforms
    cl_uint numPlatforms;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkError(err, "Finding platforms");
	if (numPlatforms == 0) {
		printf("Found 0 platforms!\n");
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id platforms[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, platforms, NULL);
	checkError(err, "Getting platforms");

	// Get Device
	for (cl_uint i = 0; i < numPlatforms; ++i) {
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
        if (err != CL_SUCCESS)
            continue;

        cl_device_id devices[numDevices];
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
		if (err == CL_SUCCESS) {
            device_id = devices[0];
			break;
        }
	}
	if (device_id == NULL) {
		checkError(err, "Getting device");
        return EXIT_FAILURE;
    }

	err = output_device_info(device_id);
	checkError(err, "Outputting device info");

	// Create a compute_context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	checkError(err, "Creating context");

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	checkError(err, "Creating command queue");

	// Create the compute program from the binary file
	printf("INFO : Loading %s\n", argv[1]);
	unsigned char *kernelbinary;
	int n_i;
	n_i = load_file_to_memory(argv[1], (char **) &kernelbinary);
	if (n_i < 0) {
		fprintf(stderr, "load_file_to_memory -> %d\n", n_i);
		return EXIT_FAILURE;
	}
	size_t n = n_i;
	int status;
	cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &n,
					(const unsigned char **)&kernelbinary, &status, &err);
	if (!program || status != CL_SUCCESS || err != CL_SUCCESS) {
		fprintf(stderr, "Error: Failed to create compute program from binary status:%d err:%d!\n", status, err);
		return EXIT_FAILURE;
	}

	err = clBuildProgram(program, 0, NULL,NULL,NULL,NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		fprintf(stderr, "Error: Failed to build program executable!\n%s\n", err_code(err));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
						sizeof(buffer), buffer, &len);
		fprintf(stderr, "%s\n", buffer);
		return EXIT_FAILURE;
	}

#if 0
    // Create the compute kernel from the program
	cl_kernel kernel_pi = clCreateKernel(program, "pi", &err);
	checkError(err, "Creating kernel");

    // Find kernel work-group size
    size_t work_group_size = 8;
    err = clGetKernelWorkGroupInfo(kernel_pi, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    checkError(err, "Getting kernel work group info");
    //printf("work_group_size : %d\n", work_group_size);

    size_t nwork_groups = in_nsteps/(work_group_size*niters);
    //printf("nwork_groups : %d\n", nwork_groups);
    if (nwork_groups < 1) {
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &nwork_groups, NULL);
        checkError(err, "Getting device compute unit info");
        work_group_size = in_nsteps / (nwork_groups * niters);
    }

    int nsteps = work_group_size * niters * nwork_groups;
    float step_size = 1.0f/(float)nsteps;

    float *h_psums = (float *) calloc(sizeof(float), nwork_groups);

    printf(" %ld work-groups of size %ld. %d Integration steps\n",
           nwork_groups, work_group_size, nsteps);

    cl_mem d_partial_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*nwork_groups, NULL, &err);
    checkError(err, "Creating buffer d_partial_sums");

    // Set kernel arguments
    err  = clSetKernelArg(kernel_pi, 0, sizeof(int), &niters);
    err |= clSetKernelArg(kernel_pi, 1, sizeof(float), &step_size);
    err |= clSetKernelArg(kernel_pi, 2, sizeof(float)*work_group_size, NULL);
    err |= clSetKernelArg(kernel_pi, 3, sizeof(float)*nwork_groups, &d_partial_sums);
    checkError(err, "Settin kernel args");

    size_t global = nsteps / niters;
    size_t local  = work_group_size;

    double rtime = wtime();
    err = clEnqueueNDRangeKernel(commands, kernel_pi, 1, NULL,
            &global, &local, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clEnqueueReadBuffer(commands, d_partial_sums, CL_TRUE, 0,
                              sizeof(float) * nwork_groups, h_psums,
                              0, NULL,NULL);
    checkError(err, "Reading back d_partial_sums");

    float pi_res = 0.0f;
    for (unsigned int i = 0; i < nwork_groups; i++)
        pi_res += h_psums[i];
    pi_res *= step_size;

	free(h_psums);

    rtime = wtime() - rtime;
    printf("\nThe calculation ran in %lf seconds\n", rtime);
    printf(" pi = %f for %d steps\n", pi_res, nsteps);

	clReleaseMemObject(d_partial_sums);
	clReleaseProgram(program);
	clReleaseKernel(kernel_pi);
#endif
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return EXIT_SUCCESS;
}
