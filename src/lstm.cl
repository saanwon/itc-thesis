#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

#if RNN_CELL_SIZE < 1024
#define MAX_CELL_SIZE   1024
#else
#define MAX_CELL_SIZE   RNN_CELL_SIZE
#endif

global float lstm_x_h[2*MAX_CELL_SIZE]  __attribute__((xcl_array_partition(complete,1)));
global float lstm_old_c[MAX_CELL_SIZE];
global float lstm_new_h[MAX_CELL_SIZE];
global float lstm_new_c[MAX_CELL_SIZE];

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_input(               int   flags,
                         __global const float *x,       // [cell_size]
                         __global const float *state    // [cell_size*2]
                        )
{
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
            lstm_x_h[i] = state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
            lstm_old_c[i] = state[j];
        }
    }
}

#define GATE_PARAM_SIZE         (2*RNN_CELL_SIZE+1)
__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_cell(__global const float *W)        // [cell_size, (2*cell_size+1)*4]

{
    int global_id       = get_global_id(0);
    int global_threads  = get_global_size(0);

    int start_ci = global_id       * RNN_CELL_SIZE / global_threads;
    int end_ci   = (global_id + 1) * RNN_CELL_SIZE / global_threads;

    float wloc[4][2*RNN_CELL_SIZE+1] __attribute__((xcl_array_partition(complete,2)));
    float gates[4] __attribute__((xcl_array_partition(complete,1)));

    /*loop_cell:*/for (int ci = start_ci; ci < end_ci; ci++) {
        __global const float *Wl = W + ci * (GATE_PARAM_SIZE * 4);

        __attribute__((xcl_pipeline_loop))
        for (int itr = 0, i = 0, j = 0; itr < (GATE_PARAM_SIZE * 4); itr++, j++) {
            if (j == GATE_PARAM_SIZE) {j = 0; i++;}
            wloc[i][j] = Wl[itr];
        }

        for (int gi = 0; gi < 4; gi++) {
            gates[gi] = wloc[gi][2*RNN_CELL_SIZE]; // bias
        }

        /*loop_x_h:*/for (int i = 0; i < 2*RNN_CELL_SIZE; i++) {
            __attribute__((opencl_unroll_hint))
            /*loop_gate:*/for (int gi = 0; gi < 4; gi++) {
                gates[gi] += lstm_x_h[i] * wloc[gi][i];
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
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_output(__global float *state)
{
    int j = 0;
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
}
