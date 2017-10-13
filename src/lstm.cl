#include "kerneldefs.h"

#define RG_SIZE         10
//#define RG_SIZE         1

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

#define MAX_CELL_SIZE   1500

global float lstm_x_h[2*MAX_CELL_SIZE];
global float lstm_old_c[MAX_CELL_SIZE];
global float lstm_gates[MAX_CELL_SIZE*4];

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
        __attribute__((xcl_pipeline_loop))
        for (int i = cell_size, j = 0; j < cell_size; ++i, ++j) {
            lstm_x_h[i] = state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0, j = cell_size; i < cell_size; ++i, ++j) {
            lstm_old_c[i] = state[j];
        }
    }
}

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_matrix(               int    cell_size,
                 __global const float *W)        // [cell_size, (2*cell_size+1)*4]

{
    int vector_size = 2 * cell_size;

    float gates[MAX_CELL_SIZE*4] __attribute__((xcl_array_partition(cyclic,RG_SIZE*4,1)));
    float wloc[MAX_CELL_SIZE*4]  __attribute__((xcl_array_partition(cyclic,RG_SIZE*4,1)));

    __attribute__((xcl_pipeline_loop))
    loop_gates_init: for (int i = 0; i < cell_size*4; i++) {
        gates[i] = W[i];
    }

    loop_matrix_col: for (int col = 0; col < vector_size; col++) {
        __global const float *Wl = W + (1+col) * cell_size * 4;
        __attribute__((xcl_pipeline_loop))
        loop_wloc_init: for (int i = 0; i < cell_size*4; i++) {
            wloc[i] = Wl[i];
        }

        __attribute__((xcl_pipeline_loop))
        loop_matrix_row: for (int row = 0; row < cell_size; row+=RG_SIZE) {
#if RG_SIZE > 1
            __attribute__((opencl_unroll_hint))
            loop_matrix_p: for (int i = 0; i < RG_SIZE; i++) {
                __attribute__((opencl_unroll_hint))
                loop_gates_item: for (int gi = 0; gi < 4; gi++) {
                    gates[(row+i)*4 + gi] += lstm_x_h[col] * wloc[(row+i)*4 + gi];
                }
            }
#else
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < 4; gi++) {
                gates[row*4 + gi] += lstm_x_h[col] * wloc[row*4 + gi];
            }
#endif
        }
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_out: for (int i = 0; i < cell_size*4; i++) {
        lstm_gates[i] = gates[i];
    }
}

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(           int  cell_size,
                    __global float *state) // [cell_size*2]
{
    float lstm_new_h[MAX_CELL_SIZE];
    float lstm_new_c[MAX_CELL_SIZE];

    loop_nonlinear: for (int ci = 0; ci < cell_size; ci++) {
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
    for (int i = 0, j = 0; i < cell_size; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = cell_size + 0; i < cell_size; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
}
