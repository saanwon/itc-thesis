#ifndef _KERNELDEF_H_
#define _KERNELDEF_H_

#define USE_XCL_DATAFLOW

#define NUM_RNN_LAYERS  2

//#define RNN_CELL_SIZE     1500
#define RNN_CELL_SIZE     200
//#define RNN_CELL_SIZE		10

#ifndef USE_XCL_DATAFLOW
#if RNN_CELL_SIZE <= 32
#define RNN_OCL_PIPE_DEPTH	32
#elif RNN_CELL_SIZE <= 256
#define RNN_OCL_PIPE_DEPTH	256
#elif RNN_CELL_SIZE <= 2048
#define RNN_OCL_PIPE_DEPTH	2048
#endif
#endif /* ! USE_XCL_DATAFLOW */

/*
 * FLAGS
 */
#define LSTM_FLAG_INIT_STATE	(1 << 0)

#endif /* ! _KERNELDEF_H_ */
