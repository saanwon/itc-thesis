#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <CL/opencl.h>

#include "kerneldefs.h"

#ifndef PATH_MAX
#define PATH_MAX        256
#endif

#define DEFAULT_DATA_DIR		"../../data.tiny"

#define TEST_LOOP_COUNT	(test_words_size-1)
//#define TEST_LOOP_COUNT	1000
//#define TEST_LOOP_COUNT	2

#define CALC_PERPLEXITY

#include "err_code.h"
extern int output_device_info(cl_device_id device_id);
extern double wtime();       // returns time since some fixed past point (wtime.c)

static int
load_xclbin_to_memory(const char *filename, char **result)
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


static cl_int vocab_size = 0;
static const char **vocabulary;
static int load_vocabulary(const char *fname)
{
    char buf[256];
    const char *delim = "\r\n";

    FILE *fp = fopen(fname, "rt");
    if (fp == NULL) {
        perror(fname);
        return -1;
    }
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        const char *word = strtok(buf, delim);
        vocabulary = (const char **)
            realloc(vocabulary, sizeof(char *) * (vocab_size+1));
        assert(vocabulary != NULL);
        vocabulary[vocab_size] = strdup(word);
        assert(vocabulary[vocab_size] != NULL);
        vocab_size += 1;
    }
    fclose(fp);

    printf("vocabulary size : %d\n", vocab_size);

    return 0;
}

static cl_int hidden_size;
static cl_float *embed_matrix;
static int load_embedding_params(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror(fname);
        return -1;
    }

    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        perror("fstat");
        close(fd);
        return -1;
    }

    if (statbuf.st_size % (vocab_size * sizeof(cl_float))) {
        printf("embedding parameter file is not aligned\n");
        close(fd);
        return -1;
    }

    hidden_size = statbuf.st_size / (vocab_size * sizeof(cl_float));
    printf("hidden_size : %d\n", hidden_size);
#ifdef RNN_CELL_SIZE
    if (hidden_size != RNN_CELL_SIZE) {
        printf("invalid hidden_size\n");
        return -1;
    }
#endif

    embed_matrix = (cl_float *) malloc(statbuf.st_size);
    assert(embed_matrix != NULL);
    if (read(fd, embed_matrix, statbuf.st_size) != statbuf.st_size) {
        printf("cannot read embedding parameter file\n");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

static cl_float *load_lstm_params(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror(fname);
        return NULL;
    }

    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        perror("fstat");
        close(fd);
        return NULL;
    }

    if (statbuf.st_size != (2*hidden_size+1)*4*hidden_size*sizeof(cl_float)) {
        printf("lstm parameter file is not aligned\n");
        close(fd);
        return NULL;
    }

    cl_float *params = (cl_float *) malloc(statbuf.st_size);
    assert(params != NULL);
    if (read(fd, params, statbuf.st_size) != statbuf.st_size) {
        printf("cannot read lstm parameter file\n");
        close(fd);
        return NULL;
    }

    close(fd);

    cl_float *weights = (cl_float *) malloc(statbuf.st_size);
    assert(weights != NULL);

    cl_float *w = weights;
    for (int r = 0; r < hidden_size; ++r) {
        for (int z = 0; z < 4; z++) {
            int s = z*hidden_size;
            for (int c = 0; c < (hidden_size*2+1); ++c)
                *w++ = params[s + r + hidden_size*4 * c];
        }
    }

#if 0
    double org_sum = 0., new_sum = 0.;
    for (int i = 0; i < (2*hidden_size+1)*4*hidden_size; ++i) {
        org_sum += params[i];
        new_sum += weights[i];
    }
    printf("sum of lstm weights : %f -> %f\n", org_sum, new_sum);
#endif

    free(params);
    return weights;
}

static cl_float *load_softmax_params(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror(fname);
        return NULL;
    }

    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        perror("fstat");
        close(fd);
        return NULL;
    }

    if (statbuf.st_size != (hidden_size+1)*vocab_size*sizeof(cl_float)) {
        printf("lstm parameter file is not aligned\n");
        close(fd);
        return NULL;
    }

    cl_float *params = (cl_float *) malloc(statbuf.st_size);
    assert(params != NULL);
    if (read(fd, params, statbuf.st_size) != statbuf.st_size) {
        printf("cannot read lstm parameter file\n");
        close(fd);
        return NULL;
    }

    close(fd);

    cl_float *weights = (cl_float *) malloc(statbuf.st_size);
    assert(weights != NULL);

    cl_float *w = weights;
    for (int r = 0; r < vocab_size; ++r) {
        for (int c = 0; c < (hidden_size+1); ++c)
            *w++ = params[r+vocab_size * c];
    }

#if 0
    double org_sum = 0., new_sum = 0.;
    for (int i = 0; i < (hidden_size+1)*vocab_size; ++i) {
        org_sum += params[i];
        new_sum += weights[i];
    }
    printf("sum of softmax weights : %f -> %f\n", org_sum, new_sum);
#endif

    free(params);
    return weights;
}

#define NUM_RNN_LAYERS  2
static cl_float *lstm_weights[NUM_RNN_LAYERS];
static cl_float *softmax_weights;
static int load_parameters(const char *folder)
{
    int status;
    char fname[PATH_MAX];

    // load vocabulary
    snprintf(fname, PATH_MAX, "%s/vocabulary.txt", folder);
    if ((status=load_vocabulary(fname)) != 0) return status;

    // load embedding weights
    snprintf(fname, PATH_MAX, "%s/params.embedding.bin", folder);
    if ((status=load_embedding_params(fname)) != 0) return status;

    // load lstm weights
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        snprintf(fname, PATH_MAX, "%s/params.lstm%d.bin", folder, i);
        lstm_weights[i] = load_lstm_params(fname);
        if (lstm_weights[i] == NULL) return -1;
    }

    snprintf(fname, PATH_MAX, "%s/params.softmax.bin", folder);
    softmax_weights = load_softmax_params(fname);
    if (softmax_weights == NULL) return -1;

    return 0;
}

static int test_words_size = 0;
static cl_int *test_words;
static int load_test_data(const char *folder)
{
    char fname[PATH_MAX];
    snprintf(fname, PATH_MAX, "%s/ptb.test.bin", folder);

    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        perror(fname);
        return -1;
    }

    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        perror("fstat");
        close(fd);
        return -1;
    }

    if (statbuf.st_size % sizeof(cl_int)) {
        printf("test words file is not aligned\n");
        close(fd);
        return -1;
    }

    test_words_size = statbuf.st_size / sizeof(cl_int);
    printf("test words size : %d\n", test_words_size);

    test_words = (cl_int *) malloc(statbuf.st_size);
    assert(test_words != NULL);
    if (read(fd, test_words, statbuf.st_size) != statbuf.st_size) {
        printf("cannot read test words file\n");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

static int lookup_embedding(int word, cl_float embedding[])
{
    if (word < 0 || word >= vocab_size) {
        printf("word(%d) is out of range\n", word);
        return -1;
    }
    memcpy(embedding, embed_matrix + hidden_size * word,
           hidden_size * sizeof(cl_float));

    return 0;
}

static void softmax(const cl_float *h, cl_float *probs)
{
    int k = 0;
    cl_float sum = 0.;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = 0.;
        for (int j = 0; j < hidden_size; ++j)
            probs[i] += h[j] * softmax_weights[k++];
        probs[i] += softmax_weights[k++];
        probs[i] = exp(probs[i]);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i)
        probs[i] /= sum;
}

static cl_float cross_entropy(const cl_float *probs, int vocab)
{
#if 0
    cl_float sum = 0.;
    for (int i = 0; i < vocab_size; ++i) {
        if (i == vocab)
            sum += log(probs[i]);
        else
            sum += log(1-probs[i]);
    }
    return -sum;
#else
    return -log(probs[vocab]);
#endif
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage : %s <xclbin> <data folder>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *data_dir = DEFAULT_DATA_DIR;
    if (argc >= 3)
    	data_dir = argv[2];

    printf("\n\n\n================================================================================\n");

    if (load_parameters(data_dir) != 0)
        return EXIT_FAILURE;
    if (load_test_data(data_dir) != 0)
        return EXIT_FAILURE;

    cl_int          err;
    cl_device_id    device_id = NULL;		// compute device id
    cl_event        event;

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
    printf("Loading %s\n", argv[1]);
    unsigned char *xclbin;
    int n_i;
    n_i = load_xclbin_to_memory(argv[1], (char **) &xclbin);
    if (n_i < 0) {
        printf("load_xclbin_to_memory -> %d\n", n_i);
        return EXIT_FAILURE;
    }
    size_t n = n_i;
    int status;
    cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                    (const unsigned char **)&xclbin, &status, &err);
    if (!program || status != CL_SUCCESS || err != CL_SUCCESS) {
            printf("Error: Failed to create compute program from binary status:%d err:%d!\n", status, err);
            return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL,NULL,NULL,NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                        sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    cl_kernel kernel_lstm = clCreateKernel(program, "lstm", &err);
    checkError(err, "Creating lstm kernel");

    // Find kernel work-group size
    size_t work_group_size = 0;
    err = clGetKernelWorkGroupInfo(kernel_lstm, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    checkError(err, "Getting kernel work group info");
    printf("work_group_size of lstm : %lu\n", work_group_size);

    cl_float *h_x = (cl_float *) malloc(sizeof(cl_float) * hidden_size);
    assert(h_x != NULL);
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(cl_float) * hidden_size, NULL, &err);
    checkError(err, "Creating buffer d_x");

    cl_float *h_h[NUM_RNN_LAYERS];
    cl_float *h_c[NUM_RNN_LAYERS];
    cl_mem d_h[NUM_RNN_LAYERS];
    cl_mem d_c[NUM_RNN_LAYERS];
    cl_mem d_w[NUM_RNN_LAYERS];
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        h_h[i] = (cl_float *) calloc(hidden_size, sizeof(cl_float));
        assert(h_h[i] != NULL);
        d_h[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    sizeof(cl_float) * hidden_size, NULL, &err);
        checkError(err, "Creating buffer d_h");
        err = clEnqueueWriteBuffer(commands, d_h[i], CL_TRUE, 0,
                    sizeof(cl_float) * hidden_size, h_h[i], 0, NULL, NULL);
        checkError(err, "Writing buffer d_h");

        h_c[i] = (cl_float *) calloc(hidden_size, sizeof(cl_float));
        assert(h_c[i] != NULL);
        d_c[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    sizeof(cl_float) * hidden_size, NULL, &err);
        checkError(err, "Creating buffer d_c");
        err = clEnqueueWriteBuffer(commands, d_c[i], CL_TRUE, 0,
                    sizeof(cl_float) * hidden_size, h_c[i], 0, NULL, NULL);
        checkError(err, "Writing buffer d_c");

        d_w[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(cl_float) * ((2*hidden_size+1)*4) * hidden_size,
                    NULL, &err);
        checkError(err, "Creating buffer d_w");
        err = clEnqueueWriteBuffer(commands, d_w[i], CL_TRUE, 0,
                    sizeof(cl_float) * ((2*hidden_size+1)*4) * hidden_size,
                    lstm_weights[i], 0, NULL, NULL);
        checkError(err, "Writing buffer d_w");
    }

    cl_float *probs = (cl_float *) malloc(vocab_size * sizeof(cl_float));
    assert(probs != NULL);

    size_t global[3] = {WORK_GROUP_SIZE, 1, 1};
    size_t local[3]  = {WORK_GROUP_SIZE, 1, 1};

    double rtime = wtime();

#ifdef CALC_PERPLEXITY
    cl_float costs = 0.;
#endif
    int percent = 0;
    int i;
    for (i = 0; i < TEST_LOOP_COUNT; ++i) {
        if (lookup_embedding(test_words[i], h_x) != 0)
            return EXIT_FAILURE;

#if 0
        printf("embedding :");
        for (int j = 0; j < hidden_size; ++j)
            printf(" %.8f", h_x[j]);
        printf("\n");
#endif

        err = clEnqueueWriteBuffer(commands, d_x, CL_TRUE, 0,
                    sizeof(cl_float) * hidden_size, h_x, 0, NULL, NULL);
        checkError(err, "Writing buffer d_x");

        for (int l = 0; l < NUM_RNN_LAYERS; ++l) {
            cl_mem *d_input = (l ? &d_h[l-1] : &d_x);

            // Set kernel arguments
            err  = clSetKernelArg(kernel_lstm, 0, sizeof(cl_int), &hidden_size);
            err |= clSetKernelArg(kernel_lstm, 1, sizeof(cl_mem), d_input);
            err |= clSetKernelArg(kernel_lstm, 2, sizeof(cl_mem), &d_h[l]);
            err |= clSetKernelArg(kernel_lstm, 3, sizeof(cl_mem), &d_c[l]);
            err |= clSetKernelArg(kernel_lstm, 4, sizeof(cl_mem), &d_w[l]);
            err |= clSetKernelArg(kernel_lstm, 5, hidden_size*sizeof(cl_float)*5, NULL);
            checkError(err, "Settin kernel args");

            err = clEnqueueNDRangeKernel(commands, kernel_lstm, 3, NULL,
                    global, local, 0, NULL, NULL/*&event*/);
            checkError(err, "Enqueueing kernel");
#if 0
            err = clWaitForEvents(1, &event);
            checkError(err, "clWaitForEvents");
#endif

#if 0
            err = clEnqueueReadBuffer(commands, d_h[l], CL_TRUE, 0,
                                      sizeof(cl_float) * hidden_size, h_h[l],
                                      0, NULL,NULL);
            checkError(err, "Reading back d_h");

            err = clEnqueueReadBuffer(commands, d_c[l], CL_TRUE, 0,
                                      sizeof(cl_float) * hidden_size, h_c[l],
                                      0, NULL,NULL);
            checkError(err, "Reading back d_c");

            printf("\nc :");
            for (int k = 0; k < hidden_size; ++k) {
                printf(" %.8f", h_c[l][k]);
            }
            printf("\n");

            printf("h :");
            for (int k = 0; k < hidden_size; ++k) {
                printf(" %.8f", h_h[l][k]);
            }
            printf("\n");
#endif
        }

        int l = NUM_RNN_LAYERS-1;
        err = clEnqueueReadBuffer(commands, d_h[l], CL_TRUE, 0,
                                  sizeof(cl_float) * hidden_size, h_h[l],
                                  0, NULL, NULL);

        checkError(err, "Reading back d_h");

#ifdef CALC_PERPLEXITY
        softmax(h_h[l], probs);
        costs += cross_entropy(probs, test_words[i+1]);
#endif

        if (percent != (i * 100) / TEST_LOOP_COUNT) {
			percent = (i * 100) / TEST_LOOP_COUNT;
			printf("\r%d%%", percent);
			fflush(stdout);
        }
    }

    rtime = wtime() - rtime;
    printf("\nThe calculation ran in %lf seconds\n", rtime);
#ifdef CALC_PERPLEXITY
    printf("Cross Entropy : %.8f\n", costs);
    printf("Perplexity : %.8f\n", exp(costs / i));
#endif

    clReleaseMemObject(d_x);
    free(h_x);
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        clReleaseMemObject(d_h[i]);
        clReleaseMemObject(d_c[i]);
        clReleaseMemObject(d_w[i]);
        free(h_h[i]);
        free(h_c[i]);
        free(lstm_weights[i]);
    }

    clReleaseProgram(program);
    clReleaseKernel(kernel_lstm);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(embed_matrix);
    free(softmax_weights);
    free(test_words);

    printf("================================================================================\n\n\n\n");

    return EXIT_SUCCESS;
}
