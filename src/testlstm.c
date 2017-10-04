#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <CL/opencl.h>

#include "kerneldefs.h"

//#define RNN_CELL_SIZE     1500
#define RNN_CELL_SIZE     200
//#define RNN_CELL_SIZE     10

#define NUM_CELL_KERNEL 8

#define NUM_RNN_LAYERS  2

#ifndef PATH_MAX
#define PATH_MAX        256
#endif

#if RNN_CELL_SIZE == 200
#define DEFAULT_DATA_DIR		"../../data.small"
#else
#define DEFAULT_DATA_DIR		"../../data.tiny"
#endif

#define TEST_LOOP_COUNT	(test_words_size-1)
//#define TEST_LOOP_COUNT	1000
//#define TEST_LOOP_COUNT	2
//#define TEST_LOOP_COUNT	1

#define CALC_PERPLEXITY
//#define USE_PIPE_FD_FOR_CALC
//#define NO_KERNEL_CALL

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

#define LSTM_PARAM_SIZE ((2*hidden_size+1)*4*hidden_size)
#ifdef USE_XCL_DATAFLOW
static cl_float *load_lstm_params(const char *fname, cl_float *weights)
#else
static cl_float *load_lstm_params(const char *fname)
#endif
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

    if (statbuf.st_size != (unsigned)(LSTM_PARAM_SIZE * sizeof(cl_float))) {
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

#ifndef USE_XCL_DATAFLOW
    cl_float *weights = (cl_float *) malloc(statbuf.st_size);
    assert(weights != NULL);
#endif

    cl_float *w = weights;
    for (int r = 0; r < hidden_size; ++r) {
        for (int z = 0; z < 4; z++) {
            int s = z*hidden_size;
            for (int c = 0; c < (hidden_size*2+1); ++c)
#if 0
                *w++ = params[s + r + hidden_size*4 * c];
#else
                w[(r*(hidden_size*2+1)+c)*4+z] = params[s + r + hidden_size*4 * c];
#endif
        }
    }

#if 0
    double org_sum = 0., new_sum = 0.;
    for (int i = 0; i < LSTM_PARAM_SIZE; ++i) {
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

    if (statbuf.st_size != (unsigned)((hidden_size+1)*vocab_size*sizeof(cl_float))) {
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

#ifdef USE_XCL_DATAFLOW
static cl_float *lstm_weights;
#else
static cl_float *lstm_weights[NUM_RNN_LAYERS];
#endif
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
#ifdef USE_XCL_DATAFLOW
    lstm_weights = (cl_float *)
            malloc(NUM_RNN_LAYERS * LSTM_PARAM_SIZE * sizeof(cl_float));
#endif /* USE_XCL_DATAFLOW */
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        snprintf(fname, PATH_MAX, "%s/params.lstm%d.bin", folder, i % 2); // FIXME
#ifdef USE_XCL_DATAFLOW
        if (load_lstm_params(fname, lstm_weights + i * LSTM_PARAM_SIZE) == NULL)
            return -1;
#else /* USE_XCL_DATAFLOW */
        lstm_weights[i] = load_lstm_params(fname);
        if (lstm_weights[i] == NULL) return -1;
#endif /* USE_XCL_DATAFLOW */
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

static cl_float *probs;
struct output_cb {
	int 	idx;
	float	*output;
};
#ifdef CALC_PERPLEXITY
static cl_float costs = 0.;

static pthread_t outputThread;
#ifdef USE_PIPE_FD_FOR_CALC
static int outputPipeFd[2] = {-1, -1};
#else
static pthread_cond_t outcb_buf_cond;
static pthread_mutex_t outcb_buf_mutex;
//static int outcb_thread_quit = 0;
static int outcb_buf_w = 0;
static int outcb_buf_size[2];
static int outcb_buf_len[2];
static int outcb_buf_total;
static struct output_cb **outcb_buf[2];
#endif
static void *cross_entropy_proc(void *arg)
{
#ifdef USE_PIPE_FD_FOR_CALC
    struct output_cb *pcb;
    while (read(outputPipeFd[0], &pcb, sizeof(struct output_cb *)) == sizeof(struct output_cb *) &&
           pcb != NULL)
    {
        softmax(pcb->output, probs);
        costs += cross_entropy(probs, test_words[pcb->idx+1]);

        free(pcb->output);
        free(pcb);
    }
#else
    pthread_mutex_lock(&outcb_buf_mutex);
    while (outcb_buf_total < TEST_LOOP_COUNT) {
        if (outcb_buf_size[outcb_buf_w] == 0) {
            pthread_cond_wait(&outcb_buf_cond, &outcb_buf_mutex);
            continue;
        }

        int i = outcb_buf_w;
        outcb_buf_w = !outcb_buf_w;

        pthread_mutex_unlock(&outcb_buf_mutex);
        for (int j = 0; j < outcb_buf_len[i]; ++j) {
            struct output_cb *pcb = outcb_buf[i][j];

            softmax(pcb->output, probs);
            costs += cross_entropy(probs, test_words[pcb->idx+1]);

            free(pcb->output);
            free(pcb);
        }

        outcb_buf_total += outcb_buf_len[i];
        outcb_buf_len[i] = 0;
        pthread_mutex_lock(&outcb_buf_mutex);
    }
    pthread_mutex_unlock(&outcb_buf_mutex);
    printf("outcb_buf_total : %d\n", outcb_buf_total);
#endif
    return NULL;
}
#endif /* CALC_PERPLEXITY */
static void processNextOutput(struct output_cb *pcb)
{
#ifdef CALC_PERPLEXITY
#ifdef USE_PIPE_FD_FOR_CALC
    if (outputPipeFd[1] != -1)
        write(outputPipeFd[1], &pcb, sizeof(struct output_cb *));
#else
    pthread_mutex_lock(&outcb_buf_mutex);
    int i = outcb_buf_w;
    if (outcb_buf_size[i] == outcb_buf_len[i]) {
        outcb_buf_size[i] += 4096;
        outcb_buf[i] = (struct output_cb **)
                realloc(outcb_buf[i], sizeof(struct output_cb *) * outcb_buf_size[i]);
        assert(outcb_buf[i] != NULL);
    }

    int j = outcb_buf_len[i]++;
    outcb_buf[i][j] = pcb;
    pthread_cond_signal(&outcb_buf_cond);
    pthread_mutex_unlock(&outcb_buf_mutex);
#endif
#else
    free(pcb->output);
    free(pcb);
#endif
}
#ifdef CALC_PERPLEXITY
static void startCrossEntropyThread(void)
{
#ifdef USE_PIPE_FD_FOR_CALC
    if (pipe(outputPipeFd) == -1)
        perror("pipe");
#else
    pthread_mutex_init(&outcb_buf_mutex, NULL);
    pthread_cond_init(&outcb_buf_cond, NULL);
#endif
    pthread_create(&outputThread, NULL, cross_entropy_proc, NULL);
}
static void stopCrossEntropyThread(void)
{
#ifdef USE_PIPE_FD_FOR_CALC
    processNextOutput(NULL);
#else
    pthread_mutex_lock(&outcb_buf_mutex);
    //outcb_thread_quit = 1;
    pthread_cond_signal(&outcb_buf_cond);
    pthread_mutex_unlock(&outcb_buf_mutex);
#endif

    pthread_join(outputThread, NULL);

#ifdef USE_PIPE_FD_FOR_CALC
    close(outputPipeFd[0]);
    outputPipeFd[0] = -1;
    close(outputPipeFd[1]);
    outputPipeFd[1] = -1;
#endif
}
#endif /* CALC_PERPLEXITY */
#if 0
static void outputCB(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    if (event_command_exec_status != CL_COMPLETE) return;
    processNextOutput((struct output_cb *) user_data);
}
#endif

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
    cl_command_queue commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
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
    cl_kernel kernel_input = clCreateKernel(program, "lstm_input", &err);
    checkError(err, "Creating lstm kernel");

    cl_kernel kernel_cell[NUM_CELL_KERNEL];
    for (int i = 0; i < NUM_CELL_KERNEL; ++i) {
        char kernel_name[64];
        sprintf(kernel_name, "lstm_cell%d", i);
        kernel_cell[i] = clCreateKernel(program, kernel_name, &err);
        checkError(err, "Creating lstm kernel");
    }

    cl_kernel kernel_output = clCreateKernel(program, "lstm_output", &err);
    checkError(err, "Creating lstm kernel");

    cl_float *h_x = (cl_float *) malloc(sizeof(cl_float) * hidden_size);
    assert(h_x != NULL);
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(cl_float) * hidden_size, NULL, &err);
    checkError(err, "Creating buffer d_x");

    cl_mem d_s[NUM_RNN_LAYERS];
    cl_mem d_w[NUM_RNN_LAYERS];
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        d_s[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                        sizeof(cl_float) * hidden_size * 2, NULL, &err);
        checkError(err, "Creating buffer d_s");

        d_w[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(cl_float) * ((2*hidden_size+1)*4) * hidden_size,
                    lstm_weights[i], &err);
        checkError(err, "Creating buffer d_w");
    }

    probs = (cl_float *) malloc(vocab_size * sizeof(cl_float));
    assert(probs != NULL);

#ifdef CALC_PERPLEXITY
    startCrossEntropyThread();
#endif

    err  = clSetKernelArg(kernel_input,  0, sizeof(cl_int), &hidden_size);
    err |= clSetKernelArg(kernel_output, 0, sizeof(cl_int), &hidden_size);
    checkError(err, "Enqueueing input/output kernel");

    for (int i = 0; i < NUM_CELL_KERNEL; ++i) {
        int start_ci =  i    * hidden_size / NUM_CELL_KERNEL;
        int end_ci   = (i+1) * hidden_size / NUM_CELL_KERNEL;
        err  = clSetKernelArg(kernel_cell[i], 0, sizeof(cl_int), &hidden_size);
        err |= clSetKernelArg(kernel_cell[i], 1, sizeof(cl_int), &start_ci);
        err |= clSetKernelArg(kernel_cell[i], 2, sizeof(cl_int), &end_ci);
        checkError(err, "Enqueueing cell kernel");
    }

    int percent = 0;

    double rtime = wtime();
    double itime = wtime();

    for (int i = 0; i < TEST_LOOP_COUNT; ++i) {
        if (lookup_embedding(test_words[i], h_x) != 0)
            return EXIT_FAILURE;

        err = clEnqueueWriteBuffer(commands, d_x, CL_TRUE, 0,
                                sizeof(cl_float) * hidden_size, h_x,
                                0, NULL, NULL);
        checkError(err, "Writing buffer d_x");

        int flags = 0;
        if (i == 0)
            flags = LSTM_FLAG_INIT_STATE;
        for (int j = 0; j < NUM_RNN_LAYERS; ++j) {
            err  = clSetKernelArg(kernel_input, 1, sizeof(cl_int), &flags);
            err |= clSetKernelArg(kernel_input, 2, sizeof(cl_mem), j ? &d_s[j-1] : &d_x);
            err |= clSetKernelArg(kernel_input, 3, sizeof(cl_mem), &d_s[j]);
            checkError(err, "Setting input kernel args");
            err = clEnqueueTask(commands, kernel_input, 0, NULL, NULL);
            checkError(err, "Enqueueing input kernel");

            err = clFinish(commands);
            checkError(err, "Enqueueing barrier");

            for (int k = 0; k < NUM_CELL_KERNEL; ++k) {
                err |= clSetKernelArg(kernel_cell[k], 3, sizeof(cl_mem), &d_w[j]);
                checkError(err, "Setting cell kernel args");
                err = clEnqueueTask(commands, kernel_cell[k], 0, NULL, NULL);
                checkError(err, "Enqueueing cell kernel");
            }

            err = clFinish(commands);
            checkError(err, "Enqueueing barrier");

            err = clSetKernelArg(kernel_output, 1, sizeof(cl_mem), &d_s[j]);
            checkError(err, "Setting output kernel args");
            err = clEnqueueTask(commands, kernel_output, 0, NULL, NULL);
            checkError(err, "Enqueueing output kernel");

            err = clFinish(commands);
            checkError(err, "Enqueueing barrier");
        }

        struct output_cb *pcb = (struct output_cb *) malloc(sizeof(struct output_cb));
        pcb->idx = i;
        pcb->output = (float *) malloc(sizeof(cl_float) * hidden_size);
        //cl_event event_rbuf;
        err = clEnqueueReadBuffer(commands, d_s[NUM_RNN_LAYERS-1], CL_TRUE, 0,
                                  sizeof(cl_float) * hidden_size, pcb->output,
				  0, NULL, NULL/*&event_rbuf*/);
        checkError(err, "Reading back d_y");
#if 0
        clSetEventCallback(event_rbuf, CL_COMPLETE, outputCB, pcb);
#else
        processNextOutput(pcb);
#endif

        if (percent != ((i+1) * 100) / TEST_LOOP_COUNT) {
            itime = wtime() - itime;
            percent = ((i+1) * 100) / TEST_LOOP_COUNT;
            printf("\r%d%% (%.3f)", percent, itime);
            fflush(stdout);

            itime = wtime();
        }
    }

    printf("\nbefore clFinish\n");
    err = clFinish(commands);
    printf("after clFinish\n");
    checkError(err, "clFinish");

    rtime = wtime() - rtime;
    printf("\nThe calculation ran in %lf seconds\n", rtime);
#ifdef CALC_PERPLEXITY
    sleep(1);
    stopCrossEntropyThread();
    pthread_mutex_lock(&outcb_buf_mutex);
    printf("Cross Entropy : %.8f\n", costs);
    printf("Perplexity : %.8f\n", exp(costs / TEST_LOOP_COUNT));
    pthread_mutex_unlock(&outcb_buf_mutex);
#endif
    //printf("in_sum : %f, out_sum : %f\n", in_sum, out_sum);

    clReleaseMemObject(d_x);
    free(h_x);
#ifdef USE_XCL_DATAFLOW
    clReleaseMemObject(d_w);
    free(lstm_weights);
#else /* USE_XCL_DATAFLOW */
    for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        clReleaseMemObject(d_s[i]);
        clReleaseMemObject(d_w[i]);
        free(lstm_weights[i]);
    }
#endif /* USE_XCL_DATAFLOW */

    clReleaseProgram(program);
    clReleaseKernel(kernel_input);
    for (int i = 0; i < NUM_CELL_KERNEL; ++i)
        clReleaseKernel(kernel_cell[i]);
    clReleaseKernel(kernel_output);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(embed_matrix);
    free(softmax_weights);
    free(test_words);
    free(probs);

    printf("================================================================================\n\n\n\n");

    return EXIT_SUCCESS;
}
