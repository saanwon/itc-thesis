#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

static float lstm_matrix(const float *in, __global const float *W)
{
    float sum = 0.;
//    __attribute__((xcl_pipeline_loop))
    lstm_matrix: for (int z = 0; z < RNN_CELL_SIZE; ++z) {
        sum += in[z] * W[z];
    }
    return sum;
}

static float lstm_gate(const float *x, const float *h, __global const float *Wxh)
{
    float sum[2];

//    __attribute__((opencl_unroll_hint))
//    __attribute__((xcl_pipeline_loop))
    lstm_gate: for (int i = 0; i < 2; ++i) {
        sum[i] = lstm_matrix(i ? h : x, Wxh + i * RNN_CELL_SIZE);
    }
    return sum[0] + sum[1] + Wxh[2*RNN_CELL_SIZE];
}

#define G_FORGET	0
#define G_INPUT		1
#define G_OUTPUT	2
#define G_NEW_J		3

#define IDX_W_X		0
#define IDX_W_H		1
static void lstm_cell(               int   idx,
                               const float *x,   // [cell_size]
                               const float *h,   // [cell_size]
                               const float *c,   // [cell_size]
                      __global const float *W,   // [cell_size, (2*cell_size+1)*4]
                                     float *new_h,
                                     float *new_c
                     )
{
    //float It, Jt, Ft, Ot;
#define It	gates[G_INPUT]
#define Jt	gates[G_NEW_J]
#define Ft	gates[G_FORGET]
#define Ot	gates[G_OUTPUT]
    float gates[4];

    /*
    __global const float *Wix = W;
    __global const float *Wih = Wix + RNN_CELL_SIZE;

    __global const float *Wjx = Wih + RNN_CELL_SIZE + 1;
    __global const float *Wjh = Wjx + RNN_CELL_SIZE;

    __global const float *Wfx = Wjh + RNN_CELL_SIZE + 1;
    __global const float *Wfh = Wfx + RNN_CELL_SIZE;

    __global const float *Wox = Wfh + RNN_CELL_SIZE + 1;
    __global const float *Woh = Wox + RNN_CELL_SIZE;
    */
    int Widx[4][2];

#define Wix Widx[G_INPUT][IDX_W_X]
#define Wih Widx[G_INPUT][IDX_W_H]

#define Wjx Widx[G_NEW_J][IDX_W_X]
#define Wjh Widx[G_NEW_J][IDX_W_H]

#define Wfx Widx[G_FORGET][IDX_W_X]
#define Wfh Widx[G_FORGET][IDX_W_H]

#define Wox Widx[G_OUTPUT][IDX_W_X]
#define Woh Widx[G_OUTPUT][IDX_W_H]

    Wix = 0;
    Wih = Wix + RNN_CELL_SIZE;

    Wjx = Wih + RNN_CELL_SIZE + 1;
    Wjh = Wjx + RNN_CELL_SIZE;

    Wfx = Wjh + RNN_CELL_SIZE + 1;
    Wfh = Wfx + RNN_CELL_SIZE;

    Wox = Wfh + RNN_CELL_SIZE + 1;
    Woh = Wox + RNN_CELL_SIZE;


    //Ft = lstm_gate(x, h, Wfx, Wfh);
    //It = lstm_gate(x, h, Wix, Wih);
    //Ot = lstm_gate(x, h, Wox, Woh);
    //Jt = lstm_gate(x, h, Wjx, Wjh);
//    __attribute__((opencl_unroll_hint))
//    __attribute__((xcl_pipeline_loop))
    lstm_cell_matrix: for (int i = 0; i < 4; ++i) {
        gates[i] = lstm_gate(x, h, W + Widx[i][IDX_W_X]);
    }

    //Ft = act_sigm(Ft);
    //It = act_sigm(It);
    //Ot = act_sigm(Ot);
//    __attribute__((opencl_unroll_hint))
//    __attribute__((xcl_pipeline_loop))
    lstm_cell_nonl: for (int i = 0; i < 3; ++i) {
        gates[i] = act_sigm(gates[i]);
    }
    Jt = act_tanh(Jt);

    // New Cell Status
    *new_c = Ft * c[idx] + It * Jt;
    // New Hidden Status
    *new_h = Ot * act_tanh(*new_c);
}


/*
 * W : (input gate  Wix[cell_size], input gate  Wih[cell_size], input gate  Bi
 *      new input   Wjx[cell_size], new input   Wjh[cell_size], new input   Wj
 *      forget gate Wfx[cell_size], forget gate Wfh[cell_size], forget gate Wf
 *      output gate Wox[cell_size], output gate Woh[cell_size], output gate Wo)
 *     ......
 *     (cell n-1)
 */
static void lstm_layer(                  int flags,
		                const float *l_x, // [cell_size]
                       __global const float *W,   // [cell_size, (2*cell_size+1)*4]
                                      float *state
#ifdef USE_XCL_DATAFLOW
                                     ,float *new_h
#endif
                      )
{
    int i;

#ifndef USE_XCL_DATAFLOW
    float new_h[RNN_CELL_SIZE];
#endif
    float new_c[RNN_CELL_SIZE];
    float *old_h = state;
    float *old_c = old_h + RNN_CELL_SIZE;

#ifndef USE_XCL_DATAFLOW
    if (flags & LSTM_FLAG_INIT_STATE) {
//        __attribute__((xcl_pipeline_loop))
        lstm_layer_init: for (i = 0; i < RNN_CELL_SIZE; ++i) {
            old_c[i] = 0.;
            old_h[i] = 0.;
        }
    }
#endif

//    __attribute__((xcl_pipeline_loop))
    lstm_layer_cell: for (i = 0; i < RNN_CELL_SIZE; ++i) {
        lstm_cell(i, l_x, old_h, old_c,
                  W + i * ((RNN_CELL_SIZE + RNN_CELL_SIZE + 1) * 4),
                  new_h + i, new_c + i);
    }

//    __attribute__((xcl_pipeline_loop))
    lstm_layer_update: for (i = 0; i < RNN_CELL_SIZE; ++i) {
    	old_c[i] = new_c[i];
    	old_h[i] = new_h[i];
    }
}

#ifdef USE_XCL_DATAFLOW
#define LSTM_HIDDEN

float lstm_state[NUM_RNN_LAYERS * RNN_CELL_SIZE * 2];


#define LSTM_PARAM_SIZE ((2*RNN_CELL_SIZE+1)*4*RNN_CELL_SIZE)


static void read_input(__global float *in, float * buffer_in)
{
    for (int i = 0 ; i < RNN_CELL_SIZE; i++) {
        buffer_in[i] =  in[i];
    }
}
static void write_result(__global float *out, float* buffer_out)
{
    for (int i = 0 ; i < RNN_CELL_SIZE ; i++){
        out[i] = buffer_out[i];
    }
}

#ifdef LSTM_HIDDEN
void lstm_hidden(                  int flags,
                  __global const float *W, // [hidden_layer][cell_size, (2*cell_size+1)*4]
                           const float *in,
                           float       *out)
{
    if (flags & LSTM_FLAG_INIT_STATE) {
//        __attribute__((xcl_pipeline_loop))
        lstm_state_init: for (int i = 0; i < NUM_RNN_LAYERS * RNN_CELL_SIZE * 2; ++i) {
            lstm_state[i] = 0;
        }
    }

    float new_h[NUM_RNN_LAYERS-1][RNN_CELL_SIZE];

    __attribute__((xcl_pipeline_loop))
    lstm_hidden: for (int i = 0; i < NUM_RNN_LAYERS; ++i) {
        lstm_layer(flags, i ? new_h[i-1] : in,
                   W + i * LSTM_PARAM_SIZE, lstm_state + i * RNN_CELL_SIZE * 2,
                   (i < (NUM_RNN_LAYERS-1)) ? new_h[i] : out);
    }
}
#endif /* LSTM_HIDDEN */


__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
__attribute__ ((xcl_dataflow))
void lstm_network(
                                 int flags,
                  __global const float *W, // [hidden_layer][cell_size, (2*cell_size+1)*4]
                  __global const float *x,
                  __global float       *y)
{
    float input[RNN_CELL_SIZE];
#ifndef LSTM_HIDDEN
    float layer0_h[RNN_CELL_SIZE];
#endif
    float output[RNN_CELL_SIZE];

    /* input layer */
    read_input(x, input);
#ifdef LSTM_HIDDEN
    /* hidden layers */
    lstm_hidden(flags, W, input, output);
#else
    /* hidden layer 0 */
    lstm_layer(flags, input,    W, lstm_state, layer0_h);
    /* hidden layer 1 */
    lstm_layer(flags, layer0_h, W + LSTM_PARAM_SIZE, lstm_state + RNN_CELL_SIZE * 2, output);
#endif
    /* output layer */
    write_result(y, output);
}

#else /* USE_XCL_DATAFLOW */
pipe float pipe_input	__attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));
pipe float pipe_output	__attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));
pipe float pipe_layer01 __attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));

float lstm_state[NUM_RNN_LAYERS][RNN_CELL_SIZE * 2];

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_input(__global const float *x)
{
//    __attribute__((xcl_pipeline_loop))
    lstm_input: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        write_pipe_block(pipe_input, x+i);
    }
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_output(__global float *h)
{
//    __attribute__((xcl_pipeline_loop))
    lstm_output: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        read_pipe_block(pipe_output, h+i);
    }
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_layer0(                 int flags,
                          __global const float *W    // [cell_size, (2*cell_size+1)*4]
                         )
{
    float l_x[RNN_CELL_SIZE];
    float *state = lstm_state[0];

//    __attribute__((xcl_pipeline_loop))
    lstm_layer0_in: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        read_pipe_block(pipe_input, l_x+i);
    }

    lstm_layer(flags, l_x, W, state);

//    __attribute__((xcl_pipeline_loop))
    lstm_layer0_out: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        write_pipe_block(pipe_layer01, state+i);
    }
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_layer1(               int   flags,
                          __global const float *W    // [cell_size, (2*cell_size+1)*4]
                         )
{
    float l_x[RNN_CELL_SIZE];
    float *state = lstm_state[1];

//    __attribute__((xcl_pipeline_loop))
    lstm_layer1_in: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        read_pipe_block(pipe_layer01, l_x+i);
    }

    lstm_layer(flags, l_x, W, state);

//    __attribute__((xcl_pipeline_loop))
    lstm_layer1_out: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        write_pipe_block(pipe_output, state+i);
    }
}
#endif /* USE_XCL_DATAFLOW */
