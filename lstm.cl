#include "kerneldefs.h"

float sigm(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

/*
 * W : (input gate  Wix[cell_size], input gate  Wih[cell_size], input gate  Bi
 *      new input   Wjx[cell_size], new input   Wjh[cell_size], new input   Wj
 *      forget gate Wfx[cell_size], forget gate Wfh[cell_size], forget gate Wf
 *      output gate Wox[cell_size], output gate Woh[cell_size], output gate Wo)
 *     ......
 *     (cell n-1)
 */
__attribute__ ((reqd_work_group_size(RNN_CELL_SIZE, 1, 1)))
__kernel void lstm(const int   cell_size,
                   __global const float *x,   // [cell_size]
                   __global       float *h,   // [cell_size]
                   __global       float *c,   // [cell_size]
                   __global const float *W    // [cell_size, (2*cell_size+1)*4]
                   )
{
    int z;
    int idx = get_global_id(0);

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
    float It = Wih[cell_size];

    __global const float *Wjx = Wih + cell_size + 1;
    __global const float *Wjh = Wjx + cell_size;
    float Jt = Wjh[cell_size];

    __global const float *Wfx = Wjh + cell_size + 1;
    __global const float *Wfh = Wfx + cell_size;
    float Ft = Wfh[cell_size];

    __global const float *Wox = Wfh + cell_size + 1;
    __global const float *Woh = Wox + cell_size;
    float Ot = Woh[cell_size];

    // Forget Gate
    for (z = 0; z < cell_size; ++z)
        Ft += x[z] * Wfx[z];
    for (z = 0; z < cell_size; ++z)
        Ft += h[z] * Wfh[z];
    Ft = sigm(Ft);

    // Input Gate
    for (z = 0; z < cell_size; ++z)
        It += x[z] * Wix[z];
    for (z = 0; z < cell_size; ++z)
        It += h[z] * Wih[z];
    It = sigm(It);

    // New Input
    for (z = 0; z < cell_size; ++z)
        Jt += x[z] * Wjx[z];
    for (z = 0; z < cell_size; ++z)
        Jt += h[z] * Wjh[z];
    Jt = tanh(Jt);

    // Output Gate
    for (z = 0; z < cell_size; ++z)
        Ot += x[z] * Wox[z];
    for (z = 0; z < cell_size; ++z)
        Ot += h[z] * Woh[z];
    Ot = sigm(Ot);


    /*
     * Update Internal Cell status
     */
    barrier(CLK_GLOBAL_MEM_FENCE);

    // New Cell Status
    c[idx] = Ft * c[idx] + It * Jt;
    // New Hidden Status
    h[idx] = Ot * tanh(c[idx]);
}
