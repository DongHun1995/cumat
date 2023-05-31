#include <cstdio>
#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define BLOCK_SIZE 32

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K)
{  
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  
  int gj = blockIdx.x;
  int gi = blockIdx.y;

  if(gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;

  int lj = threadIdx.x;
  int li = threadIdx.y;

  __shared__ float Alocal[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Blocal[BLOCK_SIZE][BLOCK_SIZE];

  float c = 0.f;

  int A_row_index = (gi * BLOCK_SIZE + li);
  int B_col_index = (gj * BLOCK_SIZE + lj);

  for (int bk = 0; bk < K; bk += BLOCK_SIZE)
  {
    int A_col_index = bk + lj;
    Alocal[li][lj] = (A_row_index < M && A_col_index < K ) ? A[A_row_index * K + A_col_index] : 0.f;

    int B_row_index = bk + li;
    Blocal[li][lj] = (B_row_index < K && B_col_index < N) ? B[B_row_index * N + B_col_index] : 0.f;

    __syncthreads();

    for (int lk = 0; lk < BLOCK_SIZE; ++lk)
    {
      c += Alocal[li][lk] * Blocal[lk][lj];
    }
    __syncthreads();
  }
  
  if (i < M && j < N)
  {
    C[i * N + j] = c;
  }

}

#define NGPU 4
#define EVENTS_PER_GPU 1 //INCREASE as needed

static size_t Mbegin[NGPU], Mend[NGPU];
static size_t ngpu;
static size_t Node_M;
static size_t mpi_world_size;
static cudaStream_t streams[NGPU];
static cudaEvent_t events[NGPU][EVENTS_PER_GPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];
static float *Node_A, *Node_C;

void matmul_initialize(int M, int N, int K) 
{
  mpi_world_size = 4;
  Node_M = M / mpi_world_size;

  Node_A = (float *)aligned_alloc(32, Node_M * K * sizeof(float));
  Node_C = (float *)aligned_alloc(32, Node_M * N * sizeof(float));

  ngpu = 4;

  for (size_t i = 0; i < ngpu; i++)
  {
    Mbegin[i] = Node_M / ngpu * i;
    Mend[i] = Node_M / ngpu * (i + 1);
    if (i == ngpu - 1) Mend[i] = Node_M;
  }

  for (size_t i = 0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    for (int j=0; j < EVENTS_PER_GPU; j++)
    {
      CHECK_CUDA(cudaEventCreate(&events[i][j]));
    }
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }

}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) 
{
  MPI_Scatter(A, Node_M * K, MPI_FLOAT, Node_A, Node_M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void*)B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &Node_A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 32 - 1) / 32, (Mend[i] - Mbegin[i] + 32 - 1) / 32);
    matmul_kernel<<<gridDim, blockDim,0, streams[i]>>>(A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  for(size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&Node_C[Mbegin[i] * N], C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }

    MPI_Gather(Node_C, Node_M * N, MPI_FLOAT, C, Node_M * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}


void matmul_finalize() 
{
  free(Node_A);
  free(Node_C);
  for(size_t i = 0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    for (int j = 0; j < EVENTS_PER_GPU; j++)
    {
      CHECK_CUDA(cudaEventDestroy(events[i][j]));
    }
  }
}