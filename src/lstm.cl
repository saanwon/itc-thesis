#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

#if RNN_CELL_SIZE < 1024
#define MAX_CELL_SIZE   1024
#else
#define MAX_CELL_SIZE   RNN_CELL_SIZE
#endif

local float lstm_x_h[2*MAX_CELL_SIZE]  __attribute__((xcl_array_partition(complete,1)));
local float lstm_old_c[MAX_CELL_SIZE];
local float lstm_new_h[MAX_CELL_SIZE];
local float lstm_new_c[MAX_CELL_SIZE];

local float lstm_gates[MAX_CELL_SIZE][4];

void lstm_input(               int   flags,
                __global const float *x,       // [cell_size]
                __global const float *state    // [cell_size*2]
                )
{
#if 0
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i) {
        lstm_x_h[i] = x[i];
    }
#else
    event_t     evt;
    evt = async_work_group_copy(lstm_x_h, x, RNN_CELL_SIZE, 0);
#endif

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
#if 0
        int j = 0;
        __attribute__((xcl_pipeline_loop))
        for (int i = RNN_CELL_SIZE; j < RNN_CELL_SIZE; ++i, ++j) {
            lstm_x_h[i] = state[j];
        }
        __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
            lstm_old_c[i] = state[j];
        }
#else
        evt = async_work_group_copy(lstm_x_h + RNN_CELL_SIZE, state, RNN_CELL_SIZE, evt);
        evt = async_work_group_copy(lstm_old_c, state + RNN_CELL_SIZE, RNN_CELL_SIZE, evt);
#endif
    }

    wait_group_events(1, &evt);
}

static void lstm_output(__global float *state)
{
#if 0
    int j = 0;
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_h[i];
    }
    __attribute__((xcl_pipeline_loop))
    for (int i = 0; i < RNN_CELL_SIZE; ++i, ++j) {
        state[j] = lstm_new_c[i];
    }
#else
    event_t     evt;
    evt = async_work_group_copy(state, lstm_new_h, RNN_CELL_SIZE, 0);
    evt = async_work_group_copy(state + RNN_CELL_SIZE, lstm_new_c, RNN_CELL_SIZE, evt);
    wait_group_events(1, &evt);
#endif
}


static void lstm_cell(               int    ci,
                      __global const float *W)
{
#define GATE_PARAM_SIZE         (2*RNN_CELL_SIZE+1)
#define PARALLEL_PARTITION      10

    float gates[PARALLEL_PARTITION][4];
    local float wloc[GATE_PARAM_SIZE*4] __attribute__((xcl_array_partition(cyclic,4,1)));

    __global const float *Wl = W + ci * (GATE_PARAM_SIZE * 4);

    event_t     evt;
    evt = async_work_group_copy(wloc, Wl, (2*RNN_CELL_SIZE+1)*4, 0);
    wait_group_events(1, &evt);

    gates[0][0] = wloc[2*RNN_CELL_SIZE*4+0];
    gates[0][1] = wloc[2*RNN_CELL_SIZE*4+1];
    gates[0][2] = wloc[2*RNN_CELL_SIZE*4+2];
    gates[0][3] = wloc[2*RNN_CELL_SIZE*4+3];

    __attribute__((xcl_pipeline_loop))
    for (int i = 1; i < PARALLEL_PARTITION; i++) {
        gates[i][0] = 0;
        gates[i][1] = 0;
        gates[i][2] = 0;
        gates[i][3] = 0;
    }

    __attribute__((xcl_pipeline_loop))
    loop_gates_sum: for (int z = 0; z < (2*RNN_CELL_SIZE)/PARALLEL_PARTITION; z++) {
        __attribute__((opencl_unroll_hint))
        loop_parallel: for (int p = 0; p < PARALLEL_PARTITION; p++) {
            int i = z * PARALLEL_PARTITION + p;
            int j = (i << 2);
            float x_h = lstm_x_h[i];
#if 0
            __attribute__((opencl_unroll_hint))
            loop_gates_item: for (int gi = 0; gi < 4; gi++, j++) {
                gates[p][gi] += x_h * wloc[j];
            }
#else
            gates[p][0] += x_h * wloc[j  ];
            gates[p][1] += x_h * wloc[j|1];
            gates[p][2] += x_h * wloc[j|2];
            gates[p][3] += x_h * wloc[j|3];
#endif
        }
    }

    __attribute__((xcl_pipeline_loop))
    for (int i = 1; i < PARALLEL_PARTITION; i++) {
        gates[0][0] += gates[i][0];
        gates[0][1] += gates[i][1];
        gates[0][2] += gates[i][2];
        gates[0][3] += gates[i][3];
    }

    lstm_gates[ci][0] = gates[0][0];
    lstm_gates[ci][1] = gates[0][1];
    lstm_gates[ci][2] = gates[0][2];
    lstm_gates[ci][3] = gates[0][3];
}

static void lstm_state_update(void)
{
    for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
        #define It      lstm_gates[ci][0]
        #define Jt      lstm_gates[ci][1]
        #define Ft      lstm_gates[ci][2]
        #define Ot      lstm_gates[ci][3]

        It = act_sigm(It);
        Ft = act_sigm(Ft);
        Ot = act_sigm(Ot);
        Jt = act_tanh(Jt);

        // New Cell Status
        lstm_new_c[ci] = Ft * lstm_old_c[ci] + It * Jt;
        // New Hidden Status
        lstm_new_h[ci] = Ot * act_tanh(lstm_new_c[ci]);
    }
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void lstm_layer(               int    flags,
                         __global const float *x,        // [cell_size]
                         __global const float *old_state,// [cell_size*2]
                         __global const float *W,        // [cell_size, (2*cell_size+1)*4]
                         __global       float *new_state // [cell_size*2]
                         )
{
    lstm_input(flags, x, old_state);

    for (int ci = 0; ci < RNN_CELL_SIZE; ci++) {
        lstm_cell(ci, W);
    }

    lstm_state_update();
    lstm_output(new_state);
}
