#include <stdio.h>

// handle this separately
// struct hostData 

typedef struct {
	float* v0;
	float* da;
	float* S0;
	float* El;
	float* log_2vMm;
	float* na;
	float* log_2gs;
} spectralData;

typedef struct {
	//Any info needed for the Kernel that does not change 
	//during Kernel execution OR spectral iteration step

	//DLM spectral parameters:
	float v_min;
	float v_max;	//Host only
	float dv;

	// DLM sizes:
	int N_v;
	int N_wG;
	int N_wL;
	int N_wG_x_N_wL;
	int N_total;

	//Work parameters :
	int Max_lines;
	int N_lines;
	int N_points_per_block;
	int N_threads_per_block;
	int N_blocks_per_grid;
	int N_points_per_thread;
	int	Max_iterations_per_thread;

	int shared_size_floats;
} initData;


typedef struct {
	//int N_points_per_block;
	int line_offset;
	//int N_iterations;
	int iv_offset;
} blockData;

typedef struct{
	//Any info needed for the Kernel that does not change during 
	//kernel execution but MAY CHANGE during spectral iteration step

	//Pressure & temperature:
	float p;
	//float T;
	float log_p;
	float hlog_T;
	float log_rT;
	float c2T;
	float rQ;

	//Spectral parameters:
	float log_wG_min;
	float log_wL_min;
	float log_dwG;
	float log_dwL;

	//Block data:
	blockData blocks[4096];//4096
} iterData ;