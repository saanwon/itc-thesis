#include "kerneldefs.h"

#define LSTM_INPUT_SIZE         (2*RNN_CELL_SIZE)

#define ALIGNED_GATE_SIZE       ((RNN_CELL_SIZE*4+(VECTOR_SIZE-1))/(VECTOR_SIZE))


#if 1
#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#else
#define act_sigm(x)     (1.0f / (1.0f + native_exp(-(x))))
#endif
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


__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_matrix(__global const VECTOR_TYPE *W)        // [cell_size, (2*cell_size+1)*4]
{
    float x_h[LSTM_INPUT_SIZE*VECTOR_SIZE]     __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float gates[ALIGNED_GATE_SIZE*VECTOR_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    __attribute__((xcl_pipeline_loop))
    loop_x_h_assign: for (int i = 0; i < LSTM_INPUT_SIZE*VECTOR_SIZE; i++) {
        x_h[i] = lstm_x_h[i/VECTOR_SIZE];
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_init: for (int i = 0; i < ALIGNED_GATE_SIZE; i++) {
        VECTOR_TYPE v = W[i];
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < VECTOR_SIZE; j++) {
            gates[i*VECTOR_SIZE+j] = v[j];
        }
    }

#if 1
    loop_matrix_col: for (int col = 0; col < LSTM_INPUT_SIZE; col++) {
        __attribute__((xcl_pipeline_loop))
        loop_matrix_row: for (int row = 0; row < ALIGNED_GATE_SIZE; row++) {
            VECTOR_TYPE wloc = W[(col+1)*ALIGNED_GATE_SIZE + row];
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < VECTOR_SIZE; gi++) {
                gates[row*VECTOR_SIZE + gi] += x_h[col*VECTOR_SIZE+gi] * wloc[gi];
            }
        }
    }
#else
    loop_matrix: for (int i = 0, row = 0, col = 0; i < LSTM_INPUT_SIZE * ALIGNED_GATE_SIZE; i++, row++) {
        VECTOR_TYPE wloc = W[ALIGNED_GATE_SIZE+i];
        if (row == ALIGNED_GATE_SIZE) {
            col++;
            row = 0;
        }
        __attribute__((opencl_unroll_hint))
        loop_gates_item: for (int gi = 0; gi < VECTOR_SIZE; gi++) {
            gates[row*VECTOR_SIZE + gi] += x_h[col*VECTOR_SIZE+gi] * wloc[gi];
        }
    }
#endif

    __attribute__((xcl_pipeline_loop))
    loop_gates_out: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
        lstm_gates[i] = gates[i];
    }
}

#if 0
#define ALIGNED_CELL_SIZE    (((RNN_CELL_SIZE+(VECTOR_SIZE-1))/VECTOR_SIZE)*VECTOR_SIZE)

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(__global float *state) // [cell_size*2]
{
    float gates[ALIGNED_GATE_SIZE*VECTOR_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE*4,1)));
    float old_c[RNN_CELL_SIZE]          __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float lstm_new_h[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float lstm_new_c[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    __attribute__((xcl_pipeline_loop))
    loop_old_c: for (int i = 0; i < RNN_CELL_SIZE; i++) {
        old_c[i] = lstm_old_c[i];
    }

#if 1
    __attribute__((xcl_pipeline_loop))
    loop_gates_in: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
        gates[i] = lstm_gates[i];
    }

    //__attribute__((opencl_unroll_hint(VECTOR_SIZE)))
    loop_nonlinear_a: for (int i = 0; i < ALIGNED_CELL_SIZE/2; i++) {
        __attribute__((opencl_unroll_hint))
        loop_nonlinear_p: for (int j = 0; j < 2; j++) {
#define It      gates[(i*2+j)*4+0]
#define Jt      gates[(i*2+j)*4+1]
#define Ft      gates[(i*2+j)*4+2]
#define Ot      gates[(i*2+j)*4+3]

            It = act_sigm(It);
            Ft = act_sigm(Ft);
            Ot = act_sigm(Ot);
            Jt = act_tanh(Jt);

            // New Cell Status
            lstm_new_c[(i*2+j)] = Ft * old_c[(i*2+j)] + It * Jt;
            // New Hidden Status
            lstm_new_h[(i*2+j)] = Ot * act_tanh(lstm_new_c[(i*2+j)]);
        }
    }
#else
    __attribute__((xcl_pipeline_loop))
    loop_gates_in: for (int i = 0; i < RNN_CELL_SIZE*4; i++) {
        int j = (i & (~3));
        if ((i%4) == 1) j += 3;
        if ((i%4) == 2) j += 1;
        if ((i%4) == 3) j += 2;
        gates[j] = lstm_gates[i];
    }

#define It      gates[ci*4+0]
#define Ft      gates[ci*4+1]
#define Ot      gates[ci*4+2]
#define Jt      gates[ci*4+3]

    __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
    loop_gate_i: for (int i = 0; i < ALIGNED_CELL_SIZE * 3; i++) {
        gates[(i/3)*4+i%3] = act_sigm(gates[(i/3)*4+i%3]);
    }
    __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
    loop_gate_f: for (int ci = 0; ci < ALIGNED_CELL_SIZE; ci++) {
        Jt = act_tanh(Jt);
    }
    __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
    loop_new_c: for (int ci = 0; ci < ALIGNED_CELL_SIZE; ci++) {
        lstm_new_c[ci] = Ft * old_c[ci] + It * Jt;
    }
    __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
    loop_new_h: for (int ci = 0; ci < ALIGNED_CELL_SIZE; ci++) {
        lstm_new_h[ci] = Ot * act_tanh(lstm_new_c[ci]);
    }
#endif

    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = RNN_CELL_SIZE + 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
}
#else

#define CU_CELL_SIZE    ((RNN_CELL_SIZE+(NUM_COMPUTE_UNITS-1))/NUM_COMPUTE_UNITS)

__kernel
__attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_nonlinear(__global float *state) // [cell_size*2]
{
    int global_id       = get_global_id(0);
    int global_threads  = get_global_size(0);

    int start_ci = global_id       * RNN_CELL_SIZE / global_threads;
    int end_ci   = (global_id + 1) * RNN_CELL_SIZE / global_threads;
    int num_ci = end_ci - start_ci;

    float gates[CU_CELL_SIZE*4] __attribute__((xcl_array_partition(cyclic,4,1)));
    float new_h[CU_CELL_SIZE];
    float new_c[CU_CELL_SIZE];

    __attribute__((xcl_pipeline_loop))
    loop_old_c: for (int i = 0, j = start_ci; i < num_ci; i++, j++) {
        new_c[i] = lstm_old_c[j];
    }
    __attribute__((xcl_pipeline_loop))
    loop_gates_in: for (int i = 0, j = start_ci*4; i < num_ci*4; i++, j++) {
        gates[i] = lstm_gates[j];
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

    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = start_ci; i < num_ci; ++i, ++j) {
        state[j] = new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0, j = RNN_CELL_SIZE + start_ci; i < num_ci; ++i, ++j) {
        state[j] = new_c[i];
    }
}
#endif
