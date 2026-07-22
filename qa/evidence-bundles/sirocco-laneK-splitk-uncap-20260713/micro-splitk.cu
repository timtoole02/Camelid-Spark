// Decisive de-risk for the SPLITK_MAX lever: does raising n_splits lift the split-K K-read
// bandwidth at high ctx (occupancy-bound, cheap win) or plateau (coalescing-bound, dead)?
// Lifts attn_sk_scores verbatim (with the shipped uint4 K-read), sweeps n_splits at ctx=32k.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__device__ __forceinline__ float f16_bits_to_f32(unsigned short h){ return __half2float(__ushort_as_half(h)); }

extern "C" __global__ void attn_sk_scores(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, const int* __restrict__ position_ptr,
    int max_pos, float scale, int n_splits)
{
    int position_count = position_ptr[0] + 1;
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const float* qh = q + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp * position_count / n_splits);
    int p_hi = (int)((long)(sp + 1) * position_count / n_splits);
    extern __shared__ float qsh[];
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    float local_max = -3.4e38f;
    for (int p = p_lo + tid; p < p_hi; p += blockDim.x) {
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f; int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d+0]*f16_bits_to_f32(k8[0]); dot += qsh[d+1]*f16_bits_to_f32(k8[1]);
            dot += qsh[d+2]*f16_bits_to_f32(k8[2]); dot += qsh[d+3]*f16_bits_to_f32(k8[3]);
            dot += qsh[d+4]*f16_bits_to_f32(k8[4]); dot += qsh[d+5]*f16_bits_to_f32(k8[5]);
            dot += qsh[d+6]*f16_bits_to_f32(k8[6]); dot += qsh[d+7]*f16_bits_to_f32(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d]*f16_bits_to_f32(kp[d]);
        float sc = dot*scale; scores_buf[(long)head*max_pos+p]=sc; local_max=fmaxf(local_max,sc);
    }
    __shared__ float red[1024]; red[tid]=local_max; __syncthreads();
    for (int s=blockDim.x>>1;s>0;s>>=1){ if(tid<s) red[tid]=fmaxf(red[tid],red[tid+s]); __syncthreads(); }
    if(tid==0) chunkmax_buf[(long)head*n_splits+sp]=red[0];
}

extern "C" __global__ void attn_sk_partial(
    const unsigned short* __restrict__ cache_v, float* __restrict__ scores_buf,
    const float* __restrict__ chunkmax_buf, float* __restrict__ lsum_buf,
    float* __restrict__ acc_buf, int n_heads, int n_kv_heads, int head_dim,
    const int* __restrict__ position_ptr, int max_pos, int n_splits){
    int position_count=position_ptr[0]+1; int head=blockIdx.x,sp=blockIdx.y;
    if(head>=n_heads||sp>=n_splits)return; int repeats=n_heads/n_kv_heads,kv_head=head/repeats;
    const unsigned short* vbase=cache_v+(long)kv_head*max_pos*head_dim;
    int p_lo=(int)((long)sp*position_count/n_splits),p_hi=(int)((long)(sp+1)*position_count/n_splits);
    float* sc_head=scores_buf+(long)head*max_pos; int tid=threadIdx.x;
    float gmax=-3.4e38f; for(int i=0;i<n_splits;i++)gmax=fmaxf(gmax,chunkmax_buf[(long)head*n_splits+i]);
    for(int p=p_lo+tid;p<p_hi;p+=blockDim.x)sc_head[p]=expf(sc_head[p]-gmax); __syncthreads();
    if(tid==0){float ls=0.f;for(int p=p_lo;p<p_hi;p++)ls+=sc_head[p];lsum_buf[(long)head*n_splits+sp]=ls;}
    for(int d=tid;d<head_dim;d+=blockDim.x){float a=0.f;for(int p=p_lo;p<p_hi;p++)a+=sc_head[p]*f16_bits_to_f32(vbase[(long)p*head_dim+d]);acc_buf[(((long)head*n_splits+sp)*head_dim)+d]=a;}}

double run_partial(int n_splits,int pc,int n_heads,int n_kv,int hd,int max_pos,int iters){
    unsigned short* cv; float* sb; float* cm; float* ls; float* ac; int* pos;
    CK(cudaMalloc(&cv,(size_t)n_kv*max_pos*hd*2)); CK(cudaMemset(cv,1,(size_t)n_kv*max_pos*hd*2));
    CK(cudaMalloc(&sb,(size_t)n_heads*max_pos*4)); CK(cudaMemset(sb,0,(size_t)n_heads*max_pos*4));
    CK(cudaMalloc(&cm,(size_t)n_heads*n_splits*4)); CK(cudaMemset(cm,0,(size_t)n_heads*n_splits*4));
    CK(cudaMalloc(&ls,(size_t)n_heads*n_splits*4));
    CK(cudaMalloc(&ac,(size_t)n_heads*n_splits*hd*4));
    int p=pc-1; CK(cudaMalloc(&pos,4)); CK(cudaMemcpy(pos,&p,4,cudaMemcpyHostToDevice));
    dim3 grid(n_heads,n_splits,1); int block=256; double vbytes=(double)n_heads*pc*hd*2;
    for(int i=0;i<10;i++) attn_sk_partial<<<grid,block>>>(cv,sb,cm,ls,ac,n_heads,n_kv,hd,pos,max_pos,n_splits);
    CK(cudaDeviceSynchronize()); cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));std::vector<double>ts;
    for(int i=0;i<iters;i++){CK(cudaEventRecord(a));attn_sk_partial<<<grid,block>>>(cv,sb,cm,ls,ac,n_heads,n_kv,hd,pos,max_pos,n_splits);
        CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));ts.push_back(vbytes/(ms/1e3)/1e9);}
    std::sort(ts.begin(),ts.end()); cudaFree(cv);cudaFree(sb);cudaFree(cm);cudaFree(ls);cudaFree(ac);cudaFree(pos); return ts[ts.size()/2];
}
double run(int n_splits,int pc,int n_heads,int n_kv,int hd,int max_pos,int iters){
    float* q; unsigned short* ck; float* sb; float* cm; int* pos;
    CK(cudaMalloc(&q,(size_t)n_heads*hd*4)); CK(cudaMemset(q,1,(size_t)n_heads*hd*4));
    CK(cudaMalloc(&ck,(size_t)n_kv*max_pos*hd*2)); CK(cudaMemset(ck,1,(size_t)n_kv*max_pos*hd*2));
    CK(cudaMalloc(&sb,(size_t)n_heads*max_pos*4));
    CK(cudaMalloc(&cm,(size_t)n_heads*n_splits*4));
    int p=pc-1; CK(cudaMalloc(&pos,4)); CK(cudaMemcpy(pos,&p,4,cudaMemcpyHostToDevice));
    dim3 grid(n_heads,n_splits,1); int block=256; size_t sh=hd*4; float scale=0.125f;
    for(int i=0;i<10;i++) attn_sk_scores<<<grid,block,sh>>>(q,ck,sb,cm,n_heads,n_kv,hd,pos,max_pos,scale,n_splits);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); std::vector<double> ts;
    // K bytes logically read = n_heads * pc * head_dim * 2 (each query head reads all pc key rows)
    double kbytes=(double)n_heads*pc*hd*2;
    for(int i=0;i<iters;i++){ CK(cudaEventRecord(a));
        attn_sk_scores<<<grid,block,sh>>>(q,ck,sb,cm,n_heads,n_kv,hd,pos,max_pos,scale,n_splits);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b));
        ts.push_back(kbytes/(ms/1e3)/1e9); }
    std::sort(ts.begin(),ts.end());
    cudaFree(q);cudaFree(ck);cudaFree(sb);cudaFree(cm);cudaFree(pos);
    return ts[ts.size()/2];
}
int main(int argc,char**argv){
    int pc=argc>1?atoi(argv[1]):32768; int n_heads=32,n_kv=8,hd=64,max_pos=pc,iters=100;
    printf("attn_sk_scores K-read @ ctx=%d (n_heads=%d n_kv=%d hd=%d), K logical=%.1f MB. peak~271 GB/s\n",
        pc,n_heads,n_kv,hd,(double)n_heads*pc*hd*2/1e6);
    printf("  current cap: n_splits = clamp(ceil(%d/256),2,16) = %d (grid %dx%d=%d blocks)\n",
        pc,std::min((pc+255)/256,16),n_heads,std::min((pc+255)/256,16),n_heads*std::min((pc+255)/256,16));
    printf("--- attn_sk_scores (K read, 256 threads/block active) ---\n");
    for(int ns : {16,32,64,128,256}){
        double gb=run(ns,pc,n_heads,n_kv,hd,max_pos,iters);
        printf("  n_splits=%-4d (grid %dx%d=%-5d blocks): %.1f GB/s (%.0f%% peak)\n",
            ns,n_heads,ns,n_heads*ns,gb,gb/271*100);
    }
    printf("--- attn_sk_partial (V read, only head_dim=%d threads/block active) ---\n",hd);
    for(int ns : {16,32,64,128,256}){
        double gb=run_partial(ns,pc,n_heads,n_kv,hd,max_pos,iters);
        printf("  n_splits=%-4d (grid %dx%d, %d active thr/block, %d total active): %.1f GB/s (%.0f%% peak)\n",
            ns,n_heads,ns,hd,n_heads*ns*hd,gb,gb/271*100);
    }
    return 0;
}
