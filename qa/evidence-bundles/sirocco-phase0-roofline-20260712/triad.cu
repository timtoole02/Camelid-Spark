// Device-to-device STREAM triad: a[i] = b[i] + q*c[i]  (2 reads + 1 write = 3 words moved)
// Measures achievable global-memory bandwidth (BW_peak) on this GPU at its current locked clock.
// Reports min / median / max over many iterations; run under the SM-clock pin.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__global__ void triad(float* __restrict__ a, const float* __restrict__ b,
                      const float* __restrict__ c, float q, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=stride) a[i] = b[i] + q*c[i];
}

int main(int argc, char** argv){
  size_t N = (argc>1)? strtoull(argv[1],0,10) : (1ULL<<27); // 134M floats -> 512MB/array
  int iters = (argc>2)? atoi(argv[2]) : 100;
  int warmup = 20;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  double bytes = 3.0 * (double)N * sizeof(float);
  printf("device=%s  N=%zu  arrays=3x%.1fMB  moved/iter=%.1fMB  memClkMax=%d MHz  busWidth=%d bit  theo=%.1f GB/s\n",
    p.name, N, N*sizeof(float)/1e6, bytes/1e6, p.memoryClockRate/1000,
    p.memoryBusWidth, 2.0*p.memoryClockRate*1e3*(p.memoryBusWidth/8)/1e9);

  float *a,*b,*c;
  CK(cudaMalloc(&a,N*sizeof(float))); CK(cudaMalloc(&b,N*sizeof(float))); CK(cudaMalloc(&c,N*sizeof(float)));
  CK(cudaMemset(a,0,N*sizeof(float))); CK(cudaMemset(b,1,N*sizeof(float))); CK(cudaMemset(c,2,N*sizeof(float)));

  int block=256; int grid=(int)std::min<size_t>((N+block-1)/block, 65535*4);
  cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

  for(int i=0;i<warmup;i++) triad<<<grid,block>>>(a,b,c,3.0f,N);
  CK(cudaDeviceSynchronize());

  std::vector<double> gbps;
  for(int i=0;i<iters;i++){
    CK(cudaEventRecord(ev0));
    triad<<<grid,block>>>(a,b,c,3.0f,N);
    CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
    float ms; CK(cudaEventElapsedTime(&ms,ev0,ev1));
    gbps.push_back(bytes/(ms/1e3)/1e9);
  }
  std::sort(gbps.begin(),gbps.end());
  double mn=gbps.front(), mx=gbps.back(), med=gbps[gbps.size()/2];
  printf("BW_triad  min=%.1f  median=%.1f  max=%.1f  GB/s   (%d iters)\n", mn, med, mx, iters);
  printf("BW_PEAK_MEDIAN_GBPS=%.2f\n", med);
  cudaFree(a);cudaFree(b);cudaFree(c);
  return 0;
}
