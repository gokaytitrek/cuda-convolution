
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define N 256 //Default matrix size NxN
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout

__global__ void convolution(int *A, int *C)
{
	//Filter
	int filter[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };

	//Needs for row-major layout
	int cols = N + 2;
	//int i = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int threadBlockSize = (N+2)/ blockDim.x;//The amount of processing per thread

	for (int b = threadIdx.x * threadBlockSize; b < (threadIdx.x + 1) * threadBlockSize; b++){
		
		i = b;
		
		for (int j = 0; j < N + 1; j++){//columns
			
			if (0 < i && i < N + 1 && 0 < j && j < N + 1)
			{
				int value = 0;
				value = value + A(i - 1, j - 1)	*  filter[0][0];
				value = value + A(i - 1, j)		*  filter[0][1];
				value = value + A(i - 1, j + 1)	*  filter[0][2];
				value = value + A(i, j - 1)		*  filter[1][0];
				value = value + A(i, j)			*  filter[1][1];
				value = value + A(i, j + 1)		*  filter[1][2];
				value = value + A(i + 1, j - 1)	*  filter[2][0];
				value = value + A(i + 1, j)		*  filter[2][1];
				value = value + A(i + 1, j + 1)	*  filter[2][2];
				C(i, j) = value;
			}
		}
	}

}

int main(void)
{
	//Host variables
	int A[N+2][N+2] = {};//+2 for padding matrix
	int *C;
	
	//Device variables
	int *A_d = 0, *C_d = 0;

	//Needs for row-major layout
	int cols = N + 2;

	//Calculate memory size 
	int memorySize = (N + 2) * (N + 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init matrix by 0
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
			A[i][j] = 0;
		}
	}

	//Generate random values between 0 and 9
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i + 1][j + 1] = rand() % 10;
		}
	}

	C = (int *)malloc(sizeof(*C)*memorySize);

	cudaMalloc((void**)&A_d, sizeof(*A_d)*memorySize);
	cudaMalloc((void**)&C_d, sizeof(*C_d)*memorySize);

	//Copy from host to device
	cudaMemcpy(A_d, A, sizeof(*A_d)*memorySize, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	convolution << <1, 512 >> >(A_d, C_d);//Block-thread
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Copy from device to host
	cudaMemcpy(C, C_d, sizeof(*C)*memorySize, cudaMemcpyDeviceToHost);

	////Print result
	//for (int i = 0; i < N + 2; i++) {
	//	for (int j = 0; j < N + 2; j++) {
	//		printf("%d ", C(i, j));
	//	}
	//	printf("\n");
	//}

	//Free memory
	cudaFree(C_d);
	cudaFree(A_d);
	free(C);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f", milliseconds);
	return EXIT_SUCCESS;
}

