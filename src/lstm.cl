#include "kerneldefs.h"

#define RG_SIZE         16 // must be 2^n

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

global float lstm_x_h[2*RNN_CELL_SIZE];
global float lstm_old_c[RNN_CELL_SIZE];
global float lstm_gates[RNN_CELL_SIZE*4];

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_input(                int   flags,
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
        __attribute__((xcl_pipeline_loop))
        for (int i = RNN_CELL_SIZE, j = 0; j < RNN_CELL_SIZE; ++i, ++j) {
            lstm_x_h[i] = state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0, j = RNN_CELL_SIZE; i < RNN_CELL_SIZE; ++i, ++j) {
            lstm_old_c[i] = state[j];
        }
    }
}


#define ALIGNED_CELL_SIZE    (((RNN_CELL_SIZE+(RG_SIZE-1))/RG_SIZE)*RG_SIZE)

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_matrix(__global const float *W)        // [cell_size, (2*cell_size+1)*4]
{
    float x_h[2*RNN_CELL_SIZE*RG_SIZE*4]            __attribute__((xcl_array_partition(cyclic,RG_SIZE*4,1)));
    float gates[ALIGNED_CELL_SIZE*4]                __attribute__((xcl_array_partition(cyclic,RG_SIZE*4,1)));
    //float gates[ALIGNED_CELL_SIZE*4]                __attribute__((xcl_array_partition(complete,1)));
    float wloc[2*RNN_CELL_SIZE*ALIGNED_CELL_SIZE*4] __attribute__((xcl_array_partition(cyclic,RG_SIZE*4,1)));

    __attribute__((xcl_pipeline_loop))
    loop_x_h_assign: for (int i = 0; i < 2*RNN_CELL_SIZE*RG_SIZE*4; i++) {
        x_h[i] = lstm_x_h[i/(RG_SIZE*4)];
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_init: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
        gates[i] = W[i];
    }

    W += RNN_CELL_SIZE*4;
    loop_wloc_init_a: for (int col = 0, j = 0, k = 0; col < 2*RNN_CELL_SIZE;
                           col++, j+=ALIGNED_CELL_SIZE*4, k+=RNN_CELL_SIZE*4)
    {
        __attribute__((xcl_pipeline_loop))
        loop_wloc_init_p: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
            wloc[j+i] = W[k+i];
        }
    }

#if 0
    loop_matrix_col: for (int col = 0, ws = 0; col < 2*RNN_CELL_SIZE; col++, ws += ALIGNED_CELL_SIZE) {
#if 1
        __attribute__((xcl_pipeline_loop))
        loop_matrix_row: for (int row = 0; row < ALIGNED_CELL_SIZE; row+=RG_SIZE) {
            __attribute__((opencl_unroll_hint))
            loop_matrix_p: for (int i = 0; i < RG_SIZE; i++) {
                __attribute__((opencl_unroll_hint))
                loop_gates_item: for (int gi = 0; gi < 4; gi++) {
                    gates[(row+i)*4 + gi] += x_h[(col*RG_SIZE+i)*4+gi] * wloc[(ws+row+i)*4 + gi];
                }
            }
        }
#else
        __attribute__((opencl_unroll_hint(RG_SIZE)))
        loop_matrix_row: for (int row = 0; row < ALIGNED_CELL_SIZE; row++) {
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < 4; gi++) {
                gates[row*4 + gi] += x_h[(col*RG_SIZE+(row%RG_SIZE))*4+gi] * wloc[(ws+row)*4 + gi];
            }
        }
#endif
    }
#else
#if 0
    __attribute__((xcl_pipeline_loop))
    loop_matrix_a: for (int cr = 0, ri = 0, col = 0;
                        cr < 2*RNN_CELL_SIZE*ALIGNED_CELL_SIZE;
                        ri++, cr+=RG_SIZE)
    {
        if (ri == (ALIGNED_CELL_SIZE/RG_SIZE)) {
            ri = 0;
            col++;
        }
        __attribute__((opencl_unroll_hint))
        loop_matrix_p: for (int i = 0; i < RG_SIZE*4; i++) {
            gates[ri*RG_SIZE*4+i] += x_h[col*RG_SIZE*4+i] * wloc[(col*(ALIGNED_CELL_SIZE/RG_SIZE)+ri)*RG_SIZE*4 + i];
        }
    }
#else
    __attribute__((opencl_unroll_hint(RG_SIZE*4)))
    loop_multiply: for (int i = 0; i < 2*RNN_CELL_SIZE*ALIGNED_CELL_SIZE*4; i++) {
        wloc[i] *= x_h[i/(ALIGNED_CELL_SIZE*4) * (RG_SIZE*4) + i%(RG_SIZE*4)];
    }
    loop_sum_a: for (int j = 0; j < 2*RNN_CELL_SIZE; j++) {
        __attribute__((opencl_unroll_hint(RG_SIZE*4)))
        loop_sum_p: for (int i = 0; i < ALIGNED_CELL_SIZE*4; i++) {
            gates[i] += wloc[j*ALIGNED_CELL_SIZE*4+i];
        }
    }
#endif
#endif

    __attribute__((xcl_pipeline_loop))
    loop_gates_out: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
        lstm_gates[i] = gates[i];
    }
}

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(__global float *state) // [cell_size*2]
{
    float lstm_new_h[RNN_CELL_SIZE];
    float lstm_new_c[RNN_CELL_SIZE];

    loop_nonlinear: for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
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
    for (int i = 0, j = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = RNN_CELL_SIZE + 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
}
