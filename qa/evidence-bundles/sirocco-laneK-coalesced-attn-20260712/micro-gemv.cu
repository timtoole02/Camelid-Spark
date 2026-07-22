// Standalone Q8_0 GEMV microbenchmark replicating Camelid's q8_gemv access pattern.
// Sweeps the latency-hiding unroll depth U and block size to find the % of peak DRAM.
// Weight bytes read/row = blocks_per_row*(32 quants + 4 scale). One warp per output row.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<int U>
__global__ void q8_gemv_u(
    const float* __restrict__ input_scales, const signed char* __restrict__ input_quants,
    const unsigned char* __restrict__ weight_bytes, int rows, int blocks_per_row,
    float* __restrict__ output)
{
    extern __shared__ unsigned char smem[];
    signed char* s_iq = (signed char*)smem;
    float* s_is = (float*)(smem + (long)blocks_per_row * 32);
    float* terms = (float*)(smem + (long)blocks_per_row * 36);
    int tid = threadIdx.x;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x)
        ((int*)s_iq)[i] = ((const int*)input_quants)[i];
    for (int i = tid; i < blocks_per_row; i += blockDim.x) s_is[i] = input_scales[i];
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31, wpb = blockDim.x >> 5;
    int row = blockIdx.x * wpb + warp;
    float* myterms = terms + (long)warp * blocks_per_row;
    if (row < rows) {
        long total_blocks = (long)rows * blocks_per_row;
        const signed char* quants = (const signed char*)weight_bytes;
        const float* scales = (const float*)(weight_bytes + total_blocks * 32);
        long row_block0 = (long)row * blocks_per_row;
        const int4* siq = (const int4*)s_iq;
        for (int base = lane; base < blocks_per_row; base += 32 * U) {
            int4 w0[U], w1[U]; float ws[U]; int present = 0;
            #pragma unroll
            for (int u = 0; u < U; u++) { int b = base + u*32;
                if (b < blocks_per_row) {
                    const int4* wq = (const int4*)(quants + (row_block0 + b) * 32);
                    w0[u]=wq[0]; w1[u]=wq[1]; ws[u]=scales[row_block0 + b]; present|=(1<<u);
                } }
            #pragma unroll
            for (int u = 0; u < U; u++) if (present & (1<<u)) { int b = base + u*32;
                int4 i0=siq[b*2], i1=siq[b*2+1]; int s=0;
                s=__dp4a(w0[u].x,i0.x,s); s=__dp4a(w0[u].y,i0.y,s); s=__dp4a(w0[u].z,i0.z,s); s=__dp4a(w0[u].w,i0.w,s);
                s=__dp4a(w1[u].x,i1.x,s); s=__dp4a(w1[u].y,i1.y,s); s=__dp4a(w1[u].z,i1.z,s); s=__dp4a(w1[u].w,i1.w,s);
                myterms[b]=(float)s*ws[u]*s_is[b];
            }
        }
    }
    __syncwarp();
    if (row < rows && lane == 0) { float acc=0.f; for (int b=0;b<blocks_per_row;b++) acc+=myterms[b]; output[row]=acc; }
}

template<int U>
double run(int rows, int bpr, int blockdim, int iters) {
    long total_blocks=(long)rows*bpr;
    size_t wbytes = total_blocks*32 + total_blocks*4;
    unsigned char* w; float* is_; signed char* iq; float* out;
    CK(cudaMalloc(&w,wbytes)); CK(cudaMemset(w,1,wbytes));
    CK(cudaMalloc(&is_,bpr*4)); CK(cudaMemset(is_,0,bpr*4));
    CK(cudaMalloc(&iq,bpr*32)); CK(cudaMemset(iq,1,bpr*32));
    CK(cudaMalloc(&out,rows*4));
    int wpb=blockdim/32; int grid=(rows+wpb-1)/wpb;
    size_t shmem = (size_t)bpr*36 + (size_t)wpb*bpr*4;
    CK(cudaFuncSetAttribute(q8_gemv_u<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    for(int i=0;i<10;i++) q8_gemv_u<U><<<grid,blockdim,shmem>>>(is_,iq,w,rows,bpr,out);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<double> gb;
    for(int i=0;i<iters;i++){ CK(cudaEventRecord(a));
        q8_gemv_u<U><<<grid,blockdim,shmem>>>(is_,iq,w,rows,bpr,out);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b)); gb.push_back((double)wbytes/(ms/1e3)/1e9); }
    std::sort(gb.begin(),gb.end());
    cudaFree(w);cudaFree(is_);cudaFree(iq);cudaFree(out);
    return gb[gb.size()/2];
}

int main(int argc,char**argv){
    int rows = argc>1?atoi(argv[1]):8192;
    int bpr  = argc>2?atoi(argv[2]):64;
    int iters= argc>3?atoi(argv[3]):200;
    double peak=270.9;
    printf("dims rows=%d bpr=%d  weight=%.1f MB  peak=%.1f GB/s\n", rows,bpr,(double)((long)rows*bpr*36)/1e6,peak);
    for(int bd : {64,128,256}){
      printf("  blockdim=%d:", bd);
      #define T(U) { double g=run<U>(rows,bpr,bd,iters); printf("  U%d=%.0f(%.0f%%)",U,g,g/peak*100); }
      T(4) T(8) T(12) T(16)
      #undef T
      printf("\n");
    }
    return 0;
}
