#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

global float lstm_x_h[2*RNN_CELL_SIZE];

global float lstm_old_c[RNN_CELL_SIZE];
global float lstm_new_h[RNN_CELL_SIZE];
global float lstm_new_c[RNN_CELL_SIZE];

global float lstm_gates[RNN_CELL_SIZE*4];

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_input(                int   cell_size,
                                int   flags,
                __global const float *x,       // [cell_size]
                __global const float *state    // [cell_size*2]
               )
{
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < cell_size; ++i) {
        lstm_x_h[i] = x[i];
    }

    if (flags & LSTM_FLAG_INIT_STATE) {
        __attribute__((xcl_pipeline_loop))
        for (int i = 0, j = cell_size; i < cell_size; i++, j++) {
            lstm_x_h[j] = 0;
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < cell_size; ++i) {
            lstm_old_c[i] = 0;
        }
    } else {
        int j = 0;
        __attribute__((xcl_pipeline_loop))
        for (int i = cell_size; j < cell_size; ++i, ++j) {
            lstm_x_h[i] = state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < cell_size; ++i, ++j) {
            lstm_old_c[i] = state[j];
        }
    }
}

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_matrix(               int    cell_size,
                 __global const float *W)        // [cell_size, (2*cell_size+1)*4]

{
#if 0
    int global_id       = get_global_id(0);
    int global_threads  = get_global_size(0);

    int start_ci = global_id       * cell_size / global_threads;
    int end_ci   = (global_id + 1) * cell_size / global_threads;
#endif

    //float x_h[2*RNN_CELL_SIZE*4]        __attribute__((xcl_array_partition(cyclic,4,1)));
    float gates[RNN_CELL_SIZE*4]                    __attribute__((xcl_array_partition(cyclic,4,1)));
    float wloc[RNN_CELL_SIZE*(2*RNN_CELL_SIZE+1)*4] __attribute__((xcl_array_partition(cyclic,4,1)));

#if 0
    __attribute__((xcl_pipeline_loop))
    loop_x_h_assign: for (int i = 0; i < 2*cell_size*4; i++) {
        x_h[i] = lstm_x_h[i>>2];
    }
#endif

    __attribute__((xcl_pipeline_loop))
    loop_wloc_init: for (int i = 0; i < RNN_CELL_SIZE*(2*RNN_CELL_SIZE+1)*4; i++) {
        wloc[i] = W[i];
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_init: for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
        __attribute__((opencl_unroll_hint))
        for (int gi = 0; gi < 4; gi++) {
            gates[ci*4+gi] = wloc[(ci+1)*(2*RNN_CELL_SIZE+1)*4-4+gi];
        }
    }

    __attribute__((xcl_pipeline_loop))
    loop_matrix: for (int ci = 0; ci < cell_size; ci++) {
        __attribute__((xcl_pipeline_loop))
        loop_gates_sum: for (int i = 0; i < 2*cell_size; i++) {
            float x_h[4] __attribute__((xcl_array_partition(complete,1)));
            x_h[3] = x_h[2] = x_h[1] = x_h[0] = lstm_x_h[i];
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < 4; gi++) {
                //gates[ci*4+gi] += x_h[i*4+gi] * wloc[i*4+gi];
                gates[ci*4+gi] += x_h[gi] * wloc[(ci*(2*RNN_CELL_SIZE+1)+i)*4+gi];
            }
        }
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_assign: for (int i = 0; i < cell_size*4; i++) {
        lstm_gates[i] = gates[i];
    }
}

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(           int  cell_size,
                    __global float *state) // [cell_size*2]
{
    int global_id       = get_global_id(0);
    int global_threads  = get_global_size(0);

    int start_ci = global_id       * cell_size / global_threads;
    int end_ci   = (global_id + 1) * cell_size / global_threads;

    loop_nonlinear: for (int ci = start_ci; ci < end_ci; ci++) {
#define It      lstm_gates[ci*4+0]
#define Jt      lstm_gates[ci*4+1]
#define Ft      lstm_gates[ci*4+2]
#define Ot      lstm_gates[ci*4+3]

        It = act_sigm(It);
        Jt = act_tanh(Jt);
        Ft = act_sigm(Ft);
        Ot = act_sigm(Ot);

        // New Cell Status
        lstm_new_c[ci] = Ft * lstm_old_c[ci] + It * Jt;
        // New Hidden Status
        lstm_new_h[ci] = Ot * act_tanh(lstm_new_c[ci]);
    }

    __attribute__((xcl_pipeline_loop))
    for (int i = start_ci, j = start_ci; i < end_ci; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = start_ci, j = cell_size + start_ci; i < end_ci; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
}
