#ifndef _KERNELDEF_H_
#define _KERNELDEF_H_

#define NUM_RNN_LAYERS  2

//#define RNN_CELL_SIZE     200
#define RNN_CELL_SIZE		10

//#define WORK_GROUP_SIZE		RNN_CELL_SIZE
#define WORK_GROUP_SIZE		1

#define LSTM_FLAG_INIT_STATE	(1 << 0)

#endif /* ! _KERNELDEF_H_ */
