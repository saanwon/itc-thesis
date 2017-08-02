#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

void lstm_cell(               int   idx,
               __local  const float *x,   // [cell_size]
               __local        float *h,   // [cell_size]
               __local        float *c,   // [cell_size]
               __global const float *W,   // [cell_size, (2*cell_size+1)*4]
			   __local        float *new_h,
			   __local        float *new_c
              )
{
    int z;
    float It, Jt, Ft, Ot;

#if 0
    if (idx == 0) {
        float sum_x = 0., sum_h = 0., sum_c = 0.;
        for (int i = 0; i < cell_size; ++i) {
            sum_x += x[i];
            sum_h += h[i];
            sum_c += c[i];
        }
        float sum_w = 0.;
        for (int i = 0; i < ((2*cell_size+1)*4) * cell_size; ++i)
            sum_w += W[i];
        printf("idx:%3d -> sum_x:%f, sum_h:%f, sum_c:%f, sum_w:%f\n",
               idx, sum_x, sum_h, sum_c, sum_w);
    }
#endif

    W += idx * ((RNN_CELL_SIZE + RNN_CELL_SIZE + 1) * 4);

    __global const float *Wix = W;
    __global const float *Wih = Wix + RNN_CELL_SIZE;
    It = Wih[RNN_CELL_SIZE];

    __global const float *Wjx = Wih + RNN_CELL_SIZE + 1;
    __global const float *Wjh = Wjx + RNN_CELL_SIZE;
    Jt = Wjh[RNN_CELL_SIZE];

    __global const float *Wfx = Wjh + RNN_CELL_SIZE + 1;
    __global const float *Wfh = Wfx + RNN_CELL_SIZE;
    Ft = Wfh[RNN_CELL_SIZE];

    __global const float *Wox = Wfh + RNN_CELL_SIZE + 1;
    __global const float *Woh = Wox + RNN_CELL_SIZE;
    Ot = Woh[RNN_CELL_SIZE];

    // Forget Gate
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Ft += x[z] * Wfx[z];
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Ft += h[z] * Wfh[z];
    Ft = act_sigm(Ft);

    // Input Gate
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        It += x[z] * Wix[z];
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        It += h[z] * Wih[z];
    It = act_sigm(It);

    // New Input
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Jt += x[z] * Wjx[z];
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Jt += h[z] * Wjh[z];
    Jt = act_tanh(Jt);

    // Output Gate
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Ot += x[z] * Wox[z];
    for (z = 0; z < RNN_CELL_SIZE; ++z)
        Ot += h[z] * Woh[z];
    Ot = act_sigm(Ot);

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
void lstm_layer(                 int flags,
		        __local  const float *l_x, // [cell_size]
                __global const float *W,   // [cell_size, (2*cell_size+1)*4]
				__local              *state
               )
{
#if WORK_GROUP_SIZE > 1
	int idx = get_global_id(0);

	lstm_cell(idx, cell_size, x, h, c, W, new_h + idx, new_c + idx);

    /*
     * Update Internal Cell status
     */
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (idx == (cell_size-1)) {
		for (int i = 0; i < cell_size; ++i) {
			c[i] = new_c[i];
			h[i] = new_h[i];
		}
    }
#else /* WORK_GROUP_SIZE */
    int i;
    event_t evt[1];

    __local float new_h[RNN_CELL_SIZE];
    __local float new_c[RNN_CELL_SIZE];
    __local float *old_h = state;
    __local float *old_c = old_h + RNN_CELL_SIZE;

    if (flags & LSTM_FLAG_INIT_STATE) {
		__attribute__((xcl_pipeline_loop))
		for (i = 0; i < RNN_CELL_SIZE; ++i) {
			old_c[i] = 0.;
			old_h[i] = 0.;
		}
    }

    //__attribute__((opencl_unroll_hint(RNN_CELL_SIZE)))
    for (i = 0; i < RNN_CELL_SIZE; ++i)
        lstm_cell(i, l_x, old_h, old_c, W, new_h + i, new_c + i);
    for (i = 0; i < RNN_CELL_SIZE; ++i) {
    	old_c[i] = new_c[i];
    	old_h[i] = new_h[i];
    }
#endif /* WORK_GROUP_SIZE */
}

pipe float pipe_input	__attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));
pipe float pipe_output	__attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));
pipe float pipe_layer01 __attribute__((xcl_reqd_pipe_depth(RNN_OCL_PIPE_DEPTH)));

__attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void lstm_input(__global const float *x)
{
	__attribute__((xcl_pipeline_loop))
	lstm_input: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		write_pipe_block(pipe_input, x+i);
	}
}

__attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void lstm_output(__global float *h)
{
	__attribute__((xcl_pipeline_loop))
	lstm_output: for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		read_pipe_block(pipe_output, h+i);
	}
}

__local float lstm_state[NUM_RNN_LAYERS][RNN_CELL_SIZE * 2];

__attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void lstm_layer0(                 int flags,
                          __global const float *W    // [cell_size, (2*cell_size+1)*4]
                         )
{
    __local float   l_x[RNN_CELL_SIZE];

	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		read_pipe_block(pipe_input, l_x+i);
	}

	lstm_layer(flags, l_x, W, lstm_state[0]);

	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		write_pipe_block(pipe_layer01, lstm_state[0]+i);
	}
}

__attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void lstm_layer1(               int   flags,
                          __global const float *W    // [cell_size, (2*cell_size+1)*4]
                         )
{
    __local float   l_x[RNN_CELL_SIZE];

	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		read_pipe_block(pipe_layer01, l_x+i);
	}

	lstm_layer(flags, l_x, W, lstm_state[1]);

	__attribute__((xcl_pipeline_loop))
	for (int i = 0; i < RNN_CELL_SIZE; ++i) {
		write_pipe_block(pipe_output, lstm_state[1]+i);
	}
}
