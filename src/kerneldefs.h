#ifndef _KERNELDEF_H_
#define _KERNELDEF_H_

//#define NUM_COMPUTE_UNITS       1

#define VECTOR_SIZE     16
#if VECTOR_SIZE == 4
#define VECTOR_TYPE     float4
#elif VECTOR_SIZE == 8
#define VECTOR_TYPE     float8
#elif VECTOR_SIZE == 16
#define VECTOR_TYPE     float16
#else
#error "Unsupported VECTOR_SIZE"
#endif

//#define RNN_CELL_SIZE     1500
#define RNN_CELL_SIZE     200
//#define RNN_CELL_SIZE     10

/*
 * FLAGS
 */
#define LSTM_FLAG_INIT_STATE	(1 << 0)

#endif /* ! _KERNELDEF_H_ */
