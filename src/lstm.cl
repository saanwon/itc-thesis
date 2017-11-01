#include "kerneldefs.h"

#define LSTM_INPUT_SIZE         (2*RNN_CELL_SIZE)

#define ALIGNED_CELL_SIZE       (((RNN_CELL_SIZE+(VECTOR_SIZE-1))/VECTOR_SIZE)*VECTOR_SIZE)
#define ALIGNED_GATE_NUM        ((RNN_CELL_SIZE*4+(VECTOR_SIZE-1))/VECTOR_SIZE)

#if 1
#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#else
#define act_sigm(x)     (1.0f / (1.0f + native_exp(-(x))))
#endif
#define act_tanh(x)     tanh(x)

#ifdef NUM_COMPUTE_UNITS
global VECTOR_TYPE lstm_old_c[ALIGNED_CELL_SIZE/VECTOR_SIZE];
global VECTOR_TYPE lstm_gates[ALIGNED_GATE_NUM];
#endif

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_matrix(int                   flags,
                 __global const VECTOR_TYPE *x,       // [cell_size]
#ifdef NUM_COMPUTE_UNITS
                 __global const VECTOR_TYPE *state,   // [cell_size*2]
#else
                 __global       VECTOR_TYPE *state,   // [cell_size*2]
#endif
                 __global const VECTOR_TYPE *W) // [cell_size, (2*cell_size+1)*4]
{
    float lstm_x_h[2*ALIGNED_CELL_SIZE]       __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    float x_h[LSTM_INPUT_SIZE*VECTOR_SIZE]    __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float gates[ALIGNED_GATE_NUM*VECTOR_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

#ifndef NUM_COMPUTE_UNITS
    float new_h[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float new_c[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
#endif

    __attribute__((xcl_pipeline_loop))
    loop_lstm_x_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
        VECTOR_TYPE v = x[i];
        __attribute__((opencl_unroll_hint))
        loop_lstm_x_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            lstm_x_h[i*VECTOR_SIZE+j] = v[j];
        }
    }

    if (flags & LSTM_FLAG_INIT_STATE) {
        __attribute__((xcl_pipeline_loop))
        loop_lstm_x_init: for (int i = 0, j = RNN_CELL_SIZE; i < RNN_CELL_SIZE; i++, j++) {
            lstm_x_h[j] = 0;
        }

#ifdef NUM_COMPUTE_UNITS
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_v_init: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = 0.0f;
        }
        __attribute__((xcl_pipeline_loop))
        loop_oldc_init: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
            lstm_old_c[i] = v;
        }
#else
        __attribute__((xcl_pipeline_loop))
        loop_oldc_init_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
            __attribute__((opencl_unroll_hint))
            loop_oldc_init_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                new_c[i*VECTOR_SIZE+j] = 0.0f;
            }
        }
#endif
    } else {
        __attribute__((xcl_pipeline_loop))
        loop_lstm_h_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
            VECTOR_TYPE v = state[i];
            __attribute__((opencl_unroll_hint))
            loop_lstm_h_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                lstm_x_h[ALIGNED_CELL_SIZE+i*VECTOR_SIZE+j] = v[j];
            }
        }
        __attribute__((xcl_pipeline_loop))
        loop_lstm_old_c: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
            VECTOR_TYPE v = state[ALIGNED_CELL_SIZE/VECTOR_SIZE+i];
#ifdef NUM_COMPUTE_UNITS
            lstm_old_c[i] = v;
#else
            __attribute__((opencl_unroll_hint))
            loop_lstm_old_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                new_c[i*VECTOR_SIZE+j] = v[j];
            }
#endif
        }
    }

    __attribute__((xcl_pipeline_loop))
    loop_x_h_assign_a: for (int i = 0; i < LSTM_INPUT_SIZE; i++) {
        int k = i;
        if (k >= RNN_CELL_SIZE)
            k += ALIGNED_CELL_SIZE - RNN_CELL_SIZE;
#if 0
        __attribute__((xcl_pipeline_loop))
        loop_x_h_assign_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            x_h[i*VECTOR_SIZE+j] = lstm_x_h[k];
        }
#else
        x_h[i*VECTOR_SIZE+0] = lstm_x_h[k];
        x_h[i*VECTOR_SIZE+1] = lstm_x_h[k];
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 2; j++) {
            x_h[i*VECTOR_SIZE+2+j] = x_h[i*VECTOR_SIZE+j];
        }
        #if VECTOR_SIZE >= 8
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 4; j++) {
            x_h[i*VECTOR_SIZE+4+j] = x_h[i*VECTOR_SIZE+j];
        }
        #endif
        #if VECTOR_SIZE >= 16
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 8; j++) {
            x_h[i*VECTOR_SIZE+8+j] = x_h[i*VECTOR_SIZE+j];
        }
        #endif
#endif
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_init: for (int i = 0; i < ALIGNED_GATE_NUM; i++) {
        VECTOR_TYPE v = W[i];
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < VECTOR_SIZE; j++) {
            gates[i*VECTOR_SIZE+j] = v[j];
        }
    }

    loop_matrix_col: for (int col = 0; col < LSTM_INPUT_SIZE; col++) {
        __attribute__((xcl_pipeline_loop))
        loop_matrix_row: for (int row = 0; row < ALIGNED_GATE_NUM; row++) {
            VECTOR_TYPE wloc = W[(col+1)*ALIGNED_GATE_NUM + row];
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < VECTOR_SIZE; gi++) {
                gates[row*VECTOR_SIZE + gi] += x_h[col*VECTOR_SIZE+gi] * wloc[gi];
            }
        }
    }

#ifdef NUM_COMPUTE_UNITS

    __attribute__((xcl_pipeline_loop))
    loop_gates_out: for (int i = 0; i < ALIGNED_GATE_NUM; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_gates_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = gates[i*VECTOR_SIZE+j];
        }
        lstm_gates[i] = v;
    }

#else // NUM_COMPUTE_UNITS

#if 0
    printf("old_c:");
    for (int i = 0; i < RNN_CELL_SIZE; i++)
        printf(" %.4f", new_c[i]);
    printf("\n");
#endif

    loop_nonlinear: for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
#define It      gates[ci*4+0]
#define Jt      gates[ci*4+1]
#define Ft      gates[ci*4+2]
#define Ot      gates[ci*4+3]

        It = act_sigm(It);
        Ft = act_sigm(Ft);
        Ot = act_sigm(Ot);
        Jt = act_tanh(Jt);

        // New Cell Status
        new_c[ci] = Ft * new_c[ci] + It * Jt;
        // New Hidden Status
        new_h[ci] = Ot * act_tanh(new_c[ci]);
    }

#if 0
    printf("new_c:");
    for (int i = 0; i < RNN_CELL_SIZE; i++)
        printf(" %.4f", new_c[i]);
    printf("\n");
#endif

    __attribute__((xcl_pipeline_loop))
    loop_new_h_out_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_h_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_h[i*VECTOR_SIZE+j];
        }
        state[i] = v;
    }
    __attribute__((xcl_pipeline_loop))
    loop_new_c_out_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_c_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_c[i*VECTOR_SIZE+j];
        }
        state[ALIGNED_CELL_SIZE/VECTOR_SIZE+i] = v;
    }
#endif // NUM_COMPUTE_UNITS
}

#ifdef NUM_COMPUTE_UNITS
//#define CU_CELL_SIZE  ((RNN_CELL_SIZE+(NUM_COMPUTE_UNITS-1))/NUM_COMPUTE_UNITS)
#define CU_CELL_SIZE_P  ((RNN_CELL_SIZE+(NUM_COMPUTE_UNITS-1))/NUM_COMPUTE_UNITS)
#define CU_CELL_SIZE    (((CU_CELL_SIZE_P+(VECTOR_SIZE-1))/VECTOR_SIZE)*VECTOR_SIZE)

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(__global VECTOR_TYPE *state) // [cell_size*2]
{
    int global_id       = get_global_id(0);
#if 0
    int global_threads  = get_global_size(0);

    int start_ci = global_id       * RNN_CELL_SIZE / global_threads;
    int end_ci   = (global_id + 1) * RNN_CELL_SIZE / global_threads;
    int num_ci = end_ci - start_ci;
#else
    int start_ci = global_id * CU_CELL_SIZE;
    int start_vi = start_ci/VECTOR_SIZE;
    int num_ci = CU_CELL_SIZE;
    if ((start_ci+num_ci) > RNN_CELL_SIZE)
        num_ci = RNN_CELL_SIZE - start_ci;
    int num_gv = (num_ci*4 + (VECTOR_SIZE-1)) / VECTOR_SIZE;
#endif

    float gates[CU_CELL_SIZE*4] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float new_h[CU_CELL_SIZE]   __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float new_c[CU_CELL_SIZE]   __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    __attribute__((xcl_pipeline_loop))
    loop_old_c_a: for (int i = 0; i < (num_ci+VECTOR_SIZE-1)/VECTOR_SIZE; i++) {
        VECTOR_TYPE v = lstm_old_c[start_vi+i];
        __attribute__((opencl_unroll_hint))
        loop_old_c_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            new_c[i*VECTOR_SIZE+j] = v[j];
        }
    }
    __attribute__((xcl_pipeline_loop))
    loop_gates_in: for (int i = 0; i < num_gv; i++) {
        VECTOR_TYPE v = lstm_gates[start_vi*4+i];
        __attribute__((opencl_unroll_hint))
        loop_gates_in_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            gates[i*VECTOR_SIZE+j] = v[j];
        }
    }

    loop_nonlinear: for (int ci = 0; ci < num_ci; ci++) {
#define It      gates[ci*4+0]
#define Jt      gates[ci*4+1]
#define Ft      gates[ci*4+2]
#define Ot      gates[ci*4+3]

        It = act_sigm(It);
        Ft = act_sigm(Ft);
        Ot = act_sigm(Ot);
        Jt = act_tanh(Jt);

        // New Cell Status
        new_c[ci] = Ft * new_c[ci] + It * Jt;
        // New Hidden Status
        new_h[ci] = Ot * act_tanh(new_c[ci]);
    }

#if 0
    __attribute__((xcl_pipeline_loop))
    loop_new_h_out: for (int i = 0, j = start_ci; i < num_ci; ++i, ++j) {
        state[j] = new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    loop_new_c_out: for (int i = 0, j = ALIGNED_CELL_SIZE + start_ci; i < num_ci; ++i, ++j) {
        state[j] = new_c[i];
    }
#else
    __attribute__((xcl_pipeline_loop))
    loop_new_h_out_a: for (int i = 0; i < (num_ci+VECTOR_SIZE-1)/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_h_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_h[i*VECTOR_SIZE+j];
        }
        state[start_vi + i] = v;
    }
    __attribute__((xcl_pipeline_loop))
    loop_new_c_out_a: for (int i = 0; i < (num_ci+VECTOR_SIZE-1)/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_c_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_c[i*VECTOR_SIZE+j];
        }
        state[ALIGNED_CELL_SIZE/VECTOR_SIZE+start_vi + i] = v;
    }
#endif
}
#endif /* NUM_COMPUTE_UNITS */
