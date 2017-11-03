#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)


#define LSTM_INPUT_SIZE         (2*RNN_CELL_SIZE)

#define ALIGNED_CELL_SIZE       (((RNN_CELL_SIZE+(VECTOR_SIZE-1))/VECTOR_SIZE)*VECTOR_SIZE)
#define ALIGNED_GATE_NUM        ((RNN_CELL_SIZE*4+(VECTOR_SIZE-1))/VECTOR_SIZE)

static
void lstm_cell(               int   cell_size,
                              int   flags,
               __global const VECTOR_TYPE *x,         // [cell_size]
               __global const VECTOR_TYPE *old_state, // [cell_size*2]
               __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
               __global       VECTOR_TYPE *new_state) // [cell_size*2]
{
    float lstm_x_h[2*ALIGNED_CELL_SIZE]       __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    float x_h[LSTM_INPUT_SIZE*VECTOR_SIZE]    __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float gates[ALIGNED_GATE_NUM*VECTOR_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

    float new_h[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));
    float new_c[ALIGNED_CELL_SIZE] __attribute__((xcl_array_partition(cyclic,VECTOR_SIZE,1)));

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

         __attribute__((xcl_pipeline_loop))
         loop_oldc_init_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
             __attribute__((opencl_unroll_hint))
             loop_oldc_init_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                 new_c[i*VECTOR_SIZE+j] = 0.0f;
             }
         }
     } else {
         __attribute__((xcl_pipeline_loop))
         loop_lstm_h_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
             VECTOR_TYPE v = old_state[i];
             __attribute__((opencl_unroll_hint))
             loop_lstm_h_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                 lstm_x_h[ALIGNED_CELL_SIZE+i*VECTOR_SIZE+j] = v[j];
             }
         }
         __attribute__((xcl_pipeline_loop))
         loop_lstm_old_c: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; ++i) {
             VECTOR_TYPE v = old_state[ALIGNED_CELL_SIZE/VECTOR_SIZE+i];
             __attribute__((opencl_unroll_hint))
             loop_lstm_old_p: for (int j = 0; j < VECTOR_SIZE; j++) {
                 new_c[i*VECTOR_SIZE+j] = v[j];
             }
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

    __attribute__((xcl_pipeline_loop))
    loop_new_h_out_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_h_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_h[i*VECTOR_SIZE+j];
        }
        new_state[i] = v;
    }
    __attribute__((xcl_pipeline_loop))
    loop_new_c_out_a: for (int i = 0; i < ALIGNED_CELL_SIZE/VECTOR_SIZE; i++) {
        VECTOR_TYPE v;
        __attribute__((opencl_unroll_hint))
        loop_new_c_out_p: for (int j = 0; j < VECTOR_SIZE; j++) {
            v[j] = new_c[i*VECTOR_SIZE+j];
        }
        new_state[ALIGNED_CELL_SIZE/VECTOR_SIZE+i] = v;
    }
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer0(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer1(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer2(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer3(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer4(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer5(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer6(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void lstm_layer7(               int   cell_size,
                                int   flags,
                 __global const VECTOR_TYPE *x,         // [cell_size]
                 __global const VECTOR_TYPE *old_state, // [cell_size*2]
                 __global const VECTOR_TYPE *W,         // [cell_size, (2*cell_size+1)*4]
                 __global VECTOR_TYPE       *new_state) // [cell_size*2]
{
    lstm_cell(cell_size, flags, x, old_state, W, new_state);
}
