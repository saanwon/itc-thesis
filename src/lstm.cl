#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

#if RNN_CELL_SIZE < 1500
#define MAX_CELL_SIZE   1500
#else
#define MAX_CELL_SIZE   RNN_CELL_SIZE
#endif

#define GATE_PARAM_SIZE         (2*RNN_CELL_SIZE+1)

static
void lstm_cell(               int   flags,
               __global const float *x,         // [cell_size]
               __global const float *old_state, // [cell_size*2]
		       __global const float *W,         // [cell_size, (2*cell_size+1)*4]
               __global float       *new_state) // [cell_size*2]
{
	float lstm_x_h[2*MAX_CELL_SIZE]  __attribute__((xcl_array_partition(complete,1)));
	float lstm_old_c[MAX_CELL_SIZE];
	float lstm_new_h[MAX_CELL_SIZE];
	float lstm_new_c[MAX_CELL_SIZE];

    float gates[4]; //__attribute__((xcl_array_partition(complete,1)));
    float wloc[(2*MAX_CELL_SIZE+1) * 4] __attribute__((xcl_array_partition(cyclic,4,1)));

    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        lstm_x_h[i] = x[i];
    }

    if (flags & LSTM_FLAG_INIT_STATE) {
        __attribute__((xcl_pipeline_loop))
        for (int i = 0, j = RNN_CELL_SIZE; i < RNN_CELL_SIZE; i++, j++) {
            lstm_x_h[j] = 0;
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < RNN_CELL_SIZE; ++i) {
            lstm_old_c[i] = 0;
        }
    } else {
        int j = 0;
        __attribute__((xcl_pipeline_loop))
        for (int i = RNN_CELL_SIZE; j < RNN_CELL_SIZE; ++i, ++j) {
            lstm_x_h[i] = old_state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
            lstm_old_c[i] = old_state[j];
        }
    }

    loop_cell: for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
        __global const float *Wl = W + ci * (GATE_PARAM_SIZE * 4);

        __attribute__((xcl_pipeline_loop))
        loop_wloc_init: for (int i = 0; i < 2*RNN_CELL_SIZE * 4; i++) {
            wloc[i] = Wl[i];
        }
        __attribute__((xcl_pipeline_loop))
        loop_gates_init: for (int gi = 0, i = 2*RNN_CELL_SIZE*4; gi < 4; gi++, i++) {
            gates[gi] = Wl[i];
        }

        __attribute__((xcl_pipeline_loop))
        loop_gates_sum: for (int i = 0, j = 0; i < 2*RNN_CELL_SIZE; i++) {
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < 4; gi++, j++) {
                gates[gi] += lstm_x_h[i] * wloc[j];
            }
        }

#define It      gates[0]
#define Jt      gates[1]
#define Ft      gates[2]
#define Ot      gates[3]

        It = act_sigm(It);
        Jt = act_tanh(Jt);
        Ft = act_sigm(Ft);
        Ot = act_sigm(Ot);

        // New Cell Status
        lstm_new_c[ci] = Ft * lstm_old_c[ci] + It * Jt;
        // New Hidden Status
        lstm_new_h[ci] = Ot * act_tanh(lstm_new_c[ci]);
    }

    int j = 0;
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        new_state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        new_state[j] = lstm_new_c[i];
    }
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer0(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer1(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer2(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer3(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer4(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer5(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer6(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer7(               int   flags,
                 __global const float *x,         // [cell_size]
                 __global const float *old_state, // [cell_size*2]
		         __global const float *W,         // [cell_size, (2*cell_size+1)*4]
                 __global float       *new_state) // [cell_size*2]
{
    lstm_cell(flags, x, old_state, W, new_state);
}
