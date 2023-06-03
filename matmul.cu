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

#define TS 32
#define WIDTH 4

static __global__ void matmul_kernel(float4 *A, float4 *B, float4 *C, int M, int N, int K)
{  
  int A_globalRow = blockDim.y * blockIdx.y + threadIdx.y;
  int B_globalCol = blockDim.x * blockIdx.x + threadIdx.x;
  int localRow = threadIdx.y;
  int localCol = threadIdx.x;

  __shared__ float4 Alocal[TS][TS/WIDTH];
  __shared__ float4 Blocal[TS][TS/WIDTH];

  float4 inter_val = { 0.0f, 0.0f, 0.0f, 0.0f};
  
  for (int bk =0; bk < K; bk += TS)
  {
    int A_globalCol = (bk/WIDTH) + localCol;
    int B_globalRow = bk + localRow;

    Alocal[localRow][localCol] = A[A_globalRow * (K / WIDTH) + A_globalCol];
    Blocal[localRow][localCol] = B[B_globalRow * (N / WIDTH) + B_globalCol];
    
    __syncthreads();

    float4 vecA, vecB;
    float valA;
    for (int k =0; k < TS/WIDTH; k++)
    {
      vecA = Alocal[localRow][k];
      for (int w =0; w < WIDTH; w++)
      {
        vecB = Blocal[WIDTH*k + w][localCol];

        switch(w)
        {
          case 0: valA = vecA.x; break;
          case 1: valA = vecA.y; break;
          case 2: valA = vecA.z; break;
          case 3: valA = vecA.w; break;
        }

        inter_val.x += vecB.x * valA;
        inter_val.y += vecB.y * valA;
        inter_val.z += vecB.z * valA;
        inter_val.w += vecB.w * valA;
      }
    }

    __syncthreads();
  }
  
    C[A_globalRow * (N / WIDTH) + B_globalCol] = inter_val;
  
}

#define NGPU 4
static size_t Mbegin[NGPU], Mend[NGPU];
static cudaStream_t streams[NGPU];

static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

static int mpi_rank, mpi_world_size;
int node_M;

#define SLICE 4
MPI_Request reqA[SLICE], reqB, req[50], gar[10];
int reqNum;

void matmul_initialize(int M, int N, int K) 
{
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  node_M = M / SLICE / mpi_world_size;

  for (size_t i = 0; i < NGPU; i++)
  {
    Mbegin[i] = node_M / NGPU * i;
    Mend[i] = node_M / NGPU * (i + 1);
    if (i == NGPU - 1) Mend[i] = node_M;
  }

  for (size_t i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  for (size_t i =0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }

}

void matmul_slice(float *A, float *C, int M, int N, int K, int buf)
{
  if (mpi_rank != 0)
  {
    MPI_Wait(&reqA[buf], MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));    
  }

  for (int i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(TS / WIDTH, TS);
    dim3 gridDim((N + TS - 1) / TS, (Mend[i] - Mbegin[i] + TS - 1) / TS);
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(reinterpret_cast<float4*>(A_gpu[i]), reinterpret_cast<float4*>(B_gpu[i]), reinterpret_cast<float4*>(C_gpu[i]), Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  for(size_t i =0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[i] * N], C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
  }

  for (size_t i =0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }

  //MPI gather C
  if(mpi_rank == 0)
  {
    for (int i = 1; i < mpi_world_size; i++)
    {
      MPI_Irecv(&C[i * node_M * N], node_M * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req[reqNum++]);
    } 
  } else {
    MPI_Isend(C, node_M * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &gar[0]);
    MPI_Request_free(&gar[0]);
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) 
{
  reqNum = 0;

  // SEND B
  if(mpi_rank == 0)
  {
    for (int i = 1; i < mpi_world_size; i++)
    {
      MPI_Isend(B, K * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &gar[i - 1]);
      MPI_Request_free(&gar[i - 1]);

    }
  } else
  {
    MPI_Irecv((float *)B, K * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqB);
  }

  // SEND A
  for (int buf = 0; buf < SLICE; buf++)
  {
    int offset = buf * M / SLICE;

    if (mpi_rank == 0)
    {
      for (int i = 1; i < mpi_world_size; i++)
      {
        MPI_Isend(&A[offset * K + i * node_M * K], node_M * K, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &gar[i-1]);
        MPI_Request_free(&gar[i-1]);

      }
    } else
    {
      MPI_Irecv((float *)&A[offset * K], node_M * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &reqA[buf]);
    }
  }

    // node B to gpu_B
  if (mpi_rank != 0)
  {
    MPI_Wait(&reqB, MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
  }

  for (int buf =0; buf < SLICE; buf++)
  {
    int offset = buf * M / SLICE;
    matmul_slice((float *)&A[offset * K], &C[offset * N], M / SLICE, N, K, buf);
  }

  MPI_Waitall(reqNum, req, MPI_STATUS_IGNORE);

}

void matmul_finalize() 
{
  for(size_t i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
}