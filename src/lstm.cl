#include "kerneldefs.h"

#define act_sigm(x)     (1.0f / (1.0f + exp(-(x))))
#define act_tanh(x)     tanh(x)

void lstm_cell(         const int   idx,
                        const int   cell_size,
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

    W += idx * ((cell_size + cell_size + 1) * 4);

    __global const float *Wix = W;
    __global const float *Wih = Wix + cell_size;
    It = Wih[cell_size];

    __global const float *Wjx = Wih + cell_size + 1;
    __global const float *Wjh = Wjx + cell_size;
    Jt = Wjh[cell_size];

    __global const float *Wfx = Wjh + cell_size + 1;
    __global const float *Wfh = Wfx + cell_size;
    Ft = Wfh[cell_size];

    __global const float *Wox = Wfh + cell_size + 1;
    __global const float *Woh = Wox + cell_size;
    Ot = Woh[cell_size];

    // Forget Gate
    for (z = 0; z < cell_size; ++z)
        Ft += x[z] * Wfx[z];
    for (z = 0; z < cell_size; ++z)
        Ft += h[z] * Wfh[z];
    Ft = act_sigm(Ft);

    // Input Gate
    for (z = 0; z < cell_size; ++z)
        It += x[z] * Wix[z];
    for (z = 0; z < cell_size; ++z)
        It += h[z] * Wih[z];
    It = act_sigm(It);

    // New Input
    for (z = 0; z < cell_size; ++z)
        Jt += x[z] * Wjx[z];
    for (z = 0; z < cell_size; ++z)
        Jt += h[z] * Wjh[z];
    Jt = act_tanh(Jt);

    // Output Gate
    for (z = 0; z < cell_size; ++z)
        Ot += x[z] * Wox[z];
    for (z = 0; z < cell_size; ++z)
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
__attribute__ ((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void lstm(const int   cell_size,
                   __global const float *x,   // [cell_size]
                   __global       float *h,   // [cell_size]
                   __global       float *c,   // [cell_size]
                   __global const float *W,   // [cell_size, (2*cell_size+1)*4]
				   __local        float *lbuf
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
    event_t evt[3];

    __local float *new_h = lbuf;
    __local float *new_c = new_h + cell_size;
    __local float *l_x   = new_c + cell_size;
    __local float *old_h = l_x   + cell_size;
    __local float *old_c = old_h + cell_size;

    evt[0] = async_work_group_copy(l_x,   x, cell_size, 0);
    evt[1] = async_work_group_copy(old_h, h, cell_size, 0);
    evt[2] = async_work_group_copy(old_c, c, cell_size, 0);

    for (i = 0; i < cell_size; ++i)
        lstm_cell(i, cell_size, l_x, old_h, old_c, W, new_h + i, new_c + i);
    for (i = 0; i < cell_size; ++i) {
    	c[i] = new_c[i];
    	h[i] = new_h[i];
    }
#endif /* WORK_GROUP_SIZE */
}
