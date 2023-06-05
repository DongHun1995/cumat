#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                                 \
  do                                                                     \
  {                                                                      \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess)                                          \
    {                                                                    \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

//하이퍼 파라미터 설정
#define TS 64
#define WORK 4
#define WTS TS / WORK
#define NGPU 4
#define SLICE 8

__global__ void matmul_kernel(float4 *A, float4 *B, float4 *C, int M, int N, int K)
{
  int B_globalCol = WORK * blockDim.x * blockIdx.x + WORK * threadIdx.x;
  int A_globalRow = WORK * blockDim.y * blockIdx.y + threadIdx.y;
  int localCol = WORK * threadIdx.x;
  int localRow = threadIdx.y;

  // 쉐어드 메모리 올리기
  __shared__ float4 Alocal[TS][TS / WORK];
  __shared__ float4 Blocal[TS][TS / WORK];
  //백터 더하는 배열 선언
  float4 acc[WORK];
  for (int i = 0; i < WORK; i++)
  {
    acc[i] = {0.0f, 0.0f, 0.0f, 0.0f};
  }
  // 타일링 Row, col 선언
  for (int toff = 0; toff < K; toff += TS)
  {
    int tileCol = toff + localCol;
    int tileRow = toff + localRow;
    // 쉐어드 메모리 올리기
    for (int i = 0; i < WORK; i++)
    {
      Alocal[localRow + i * WTS][localCol / WORK] = A[((A_globalRow + i * WTS) * K + tileCol) / WORK];
      Blocal[localRow + i * WTS][localCol / WORK] = B[((tileRow + i * WTS) * N + B_globalCol) / WORK];
    }
    __syncthreads();
    // 열로는 백터타입, 행으로는 granulality 올리기
    float4 vector_A, vector_B;
    float val_A;
    for (int k = 0; k < TS / WORK; k++)
    {
      for (int i = 0; i < WORK; i++)
      {
        vector_A = Alocal[localRow + i * WTS][k];
        for (int w = 0; w < WORK; w++)
        {
          vector_B = Blocal[WORK * k + w][localCol / WORK];
          switch (w)
          {
          case 0: val_A = vector_A.x; break;
          case 1: val_A = vector_A.y; break;
          case 2: val_A = vector_A.z; break;
          case 3: val_A = vector_A.w; break;
          }
          acc[i].x += vector_B.x * val_A;
          acc[i].y += vector_B.y * val_A;
          acc[i].z += vector_B.z * val_A;
          acc[i].w += vector_B.w * val_A;
        }
      }
    }
    __syncthreads();
  }
  //최종 C에 더하기
  for (int i = 0; i < WORK; i++)
  {
    C[((A_globalRow + i * WTS) * N + B_globalCol) / WORK] = acc[i];
  }
}

//GPU 변수 설정
static float *gpu_A[NGPU], *gpu_B[NGPU], *gpu_C[NGPU];
static cudaStream_t up[NGPU], calc0[NGPU], calc1[NGPU], down[NGPU];
static cudaEvent_t event0[NGPU], event1[NGPU], event2[NGPU];

void matmul_initialize(int M, int N, int K)
{
  for (int i = 0; i < NGPU; i++)
  {    
    CHECK_CUDA(cudaSetDevice(i));    
    // stream 설정
    CHECK_CUDA(cudaStreamCreate(&up[i]));
    CHECK_CUDA(cudaStreamCreate(&calc0[i]));
    CHECK_CUDA(cudaStreamCreate(&calc1[i]));
    CHECK_CUDA(cudaStreamCreate(&down[i]));
    // event 설정
    CHECK_CUDA(cudaEventCreate(&event0[i]));
    CHECK_CUDA(cudaEventCreate(&event1[i]));
    CHECK_CUDA(cudaEventCreate(&event2[i]));
    // 각 gpu당 matrix memory 할당
    int gpu_M = M / NGPU;
    CHECK_CUDA(cudaMalloc(&gpu_A[i], gpu_M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_B[i], K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_C[i], gpu_M * N * sizeof(float)));
  }
}

void matmul_slice(const float *A, float *C, int M, int N, int K, int buffer)
{
  for (int i = 0; i < NGPU; i++)
  { // GPU당 슬라이스 나누기
    CHECK_CUDA(cudaSetDevice(i));
    int Mslice = M / NGPU;
    int Mbegin = Mslice * i;
    // 각 GPU에 A슬라이스 올리기
    CHECK_CUDA(cudaMemcpyAsync(gpu_A[i] + (Mslice * K) * buffer, A + Mbegin * K, Mslice * K * sizeof(float), cudaMemcpyHostToDevice, up[i]));
    CHECK_CUDA(cudaEventRecord(event0[i], up[i]));
    // 각 A슬라이스 스트림 기다리기
    CHECK_CUDA(cudaStreamWaitEvent(calc0[i], event0[i]));
    CHECK_CUDA(cudaStreamWaitEvent(calc1[i], event0[i]));
    // 각 kernel 두개로 또 나눠서 스트림 연산
    dim3 blockDim(TS / WORK, TS / WORK);
    dim3 gridDim((N + TS - 1) / TS, (Mslice / 2 + TS - 1) / TS);
    matmul_kernel<<<gridDim, blockDim, 0, calc0[i]>>>((float4 *)(gpu_A[i] + (Mslice * K) * buffer), (float4 *)gpu_B[i], (float4 *)(gpu_C[i] + (Mslice * N) * buffer), Mslice / 2, N, K);
    matmul_kernel<<<gridDim, blockDim, 0, calc1[i]>>>((float4 *)(gpu_A[i] + (Mslice * K) * buffer + (Mslice / 2 * K)), (float4 *)gpu_B[i], (float4 *)(gpu_C[i] + (Mslice * N) * buffer + (Mslice / 2 * N)), Mslice / 2, N, K);
    CHECK_CUDA(cudaEventRecord(event1[i], calc0[i]));
    CHECK_CUDA(cudaEventRecord(event2[i], calc1[i]));
    // 커널 연산 스트림 기다리기
    CHECK_CUDA(cudaStreamWaitEvent(down[i], event1[i]));
    CHECK_CUDA(cudaStreamWaitEvent(down[i], event2[i]));
    // C를 Host로 보내기
    CHECK_CUDA(cudaMemcpyAsync(C + Mbegin * N, gpu_C[i] + (Mslice * N) * buffer, Mslice * N * sizeof(float), cudaMemcpyDeviceToHost, down[i]));
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
  // B 모든 GPU에 보내 놓기
  for (int i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(gpu_B[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, up[i]));
  }
  // A를 여러개로 쪼개서 GPU로 연산하기 위한 SLICE 작업 실시
  for (int buffer = 0; buffer < SLICE; buffer++)
  {
    int offset = buffer * M / SLICE;
    matmul_slice(A + offset * K, C + offset * N, M / SLICE, N, K, buffer);
  }
  // 마지막 싱크 맞추기
  for (int i = 0; i < NGPU; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(up[i]));
    CHECK_CUDA(cudaStreamSynchronize(calc0[i]));
    CHECK_CUDA(cudaStreamSynchronize(calc1[i]));
    CHECK_CUDA(cudaStreamSynchronize(down[i]));
  }
}

void matmul_finalize()
{
  for (int i = 0; i < NGPU; i++)
  {
    // 스트림 파괴
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamDestroy(up[i]));
    CHECK_CUDA(cudaStreamDestroy(calc0[i]));
    CHECK_CUDA(cudaStreamDestroy(calc1[i]));
    CHECK_CUDA(cudaStreamDestroy(down[i]));
    // 이벤트 파괴
    CHECK_CUDA(cudaEventDestroy(event0[i]));
    CHECK_CUDA(cudaEventDestroy(event1[i]));
    CHECK_CUDA(cudaEventDestroy(event2[i]));
    // 메모리 프리작업
    CHECK_CUDA(cudaFree(gpu_A[i]));
    CHECK_CUDA(cudaFree(gpu_B[i]));
    CHECK_CUDA(cudaFree(gpu_C[i]));
  }
}