// SIROCCO Compute M-C0: does the runtime k_tokens loop force the flash per-row arrays into LOCAL
// memory (stack frame > 0), and does a compile-time k==8 unroll promote them to registers (frame 0)?
// Compile: nvcc -arch=sm_86 --fmad=false -Xptxas=-v -cubin mc0-ptxas.cu   (read "stack frame" per kernel)
#include <cuda_fp16.h>
__device__ __forceinline__ float f16_bits_to_f32(unsigned short h){ return __half2float(__ushort_as_half(h)); }
#define FLASH_MAX_BQ 16

// ---- RUNTIME k_tokens (the shipped M1 kernel; k_tokens is a param) ----
extern "C" __global__ void scores_runtime(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int k_tokens,
    int q_per_token, int max_pos, float scale, int n_splits, int position_count)
{
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp*position_count/n_splits), p_hi=(int)((long)(sp+1)*position_count/n_splits);
    extern __shared__ float qsh[]; int tid=threadIdx.x;
    for (int i=tid;i<k_tokens*head_dim;i+=blockDim.x) qsh[i]=q[(long)(i/head_dim)*q_per_token+(long)head*head_dim+(i%head_dim)];
    __syncthreads();
    int kd8=((head_dim&7)==0)?head_dim:0;
    float local_max[FLASH_MAX_BQ]; for(int t=0;t<k_tokens;t++) local_max[t]=-3.4e38f;
    for (int p=p_lo+tid;p<p_hi;p+=blockDim.x){
        const unsigned short* kp=kbase+(long)p*head_dim;
        float dot[FLASH_MAX_BQ]; for(int t=0;t<k_tokens;t++) dot[t]=0.0f;
        int d=0;
        for(;d<kd8;d+=8){ uint4 kv=*reinterpret_cast<const uint4*>(kp+d); const unsigned short* k8=reinterpret_cast<const unsigned short*>(&kv);
            float kf0=f16_bits_to_f32(k8[0]),kf1=f16_bits_to_f32(k8[1]),kf2=f16_bits_to_f32(k8[2]),kf3=f16_bits_to_f32(k8[3]);
            float kf4=f16_bits_to_f32(k8[4]),kf5=f16_bits_to_f32(k8[5]),kf6=f16_bits_to_f32(k8[6]),kf7=f16_bits_to_f32(k8[7]);
            for(int t=0;t<k_tokens;t++){ const float* qt=qsh+(long)t*head_dim;
                dot[t]+=qt[d+0]*kf0;dot[t]+=qt[d+1]*kf1;dot[t]+=qt[d+2]*kf2;dot[t]+=qt[d+3]*kf3;
                dot[t]+=qt[d+4]*kf4;dot[t]+=qt[d+5]*kf5;dot[t]+=qt[d+6]*kf6;dot[t]+=qt[d+7]*kf7; } }
        for(;d<head_dim;d++){ float kf=f16_bits_to_f32(kp[d]); for(int t=0;t<k_tokens;t++) dot[t]+=qsh[(long)t*head_dim+d]*kf; }
        for(int t=0;t<k_tokens;t++){ float sc=(p<=base_position+t)?dot[t]*scale:-3.4e38f; scores_buf[((long)t*n_heads+head)*max_pos+p]=sc; local_max[t]=fmaxf(local_max[t],sc); }
    }
    __shared__ float red[256];
    for(int t=0;t<k_tokens;t++){ red[tid]=local_max[t]; __syncthreads();
        for(int s=blockDim.x>>1;s>0;s>>=1){ if(tid<s) red[tid]=fmaxf(red[tid],red[tid+s]); __syncthreads(); }
        if(tid==0) chunkmax_buf[((long)t*n_heads+head)*n_splits+sp]=red[0]; __syncthreads(); }
}

// ---- COMPILE-TIME k==8 (KT const; loops bounded by KT so ptxas unrolls -> arrays in registers) ----
#define KT 8
extern "C" __global__ void scores_k8(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    float* __restrict__ scores_buf, float* __restrict__ chunkmax_buf,
    int n_heads, int n_kv_heads, int head_dim, int base_position,
    int q_per_token, int max_pos, float scale, int n_splits, int position_count)
{
    int head = blockIdx.x, sp = blockIdx.y;
    if (head >= n_heads || sp >= n_splits) return;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    int p_lo = (int)((long)sp*position_count/n_splits), p_hi=(int)((long)(sp+1)*position_count/n_splits);
    extern __shared__ float qsh[]; int tid=threadIdx.x;
    for (int i=tid;i<KT*head_dim;i+=blockDim.x) qsh[i]=q[(long)(i/head_dim)*q_per_token+(long)head*head_dim+(i%head_dim)];
    __syncthreads();
    int kd8=((head_dim&7)==0)?head_dim:0;
    float local_max[KT];
    #pragma unroll
    for(int t=0;t<KT;t++) local_max[t]=-3.4e38f;
    for (int p=p_lo+tid;p<p_hi;p+=blockDim.x){
        const unsigned short* kp=kbase+(long)p*head_dim;
        float dot[KT];
        #pragma unroll
        for(int t=0;t<KT;t++) dot[t]=0.0f;
        int d=0;
        for(;d<kd8;d+=8){ uint4 kv=*reinterpret_cast<const uint4*>(kp+d); const unsigned short* k8=reinterpret_cast<const unsigned short*>(&kv);
            float kf0=f16_bits_to_f32(k8[0]),kf1=f16_bits_to_f32(k8[1]),kf2=f16_bits_to_f32(k8[2]),kf3=f16_bits_to_f32(k8[3]);
            float kf4=f16_bits_to_f32(k8[4]),kf5=f16_bits_to_f32(k8[5]),kf6=f16_bits_to_f32(k8[6]),kf7=f16_bits_to_f32(k8[7]);
            #pragma unroll
            for(int t=0;t<KT;t++){ const float* qt=qsh+(long)t*head_dim;
                dot[t]+=qt[d+0]*kf0;dot[t]+=qt[d+1]*kf1;dot[t]+=qt[d+2]*kf2;dot[t]+=qt[d+3]*kf3;
                dot[t]+=qt[d+4]*kf4;dot[t]+=qt[d+5]*kf5;dot[t]+=qt[d+6]*kf6;dot[t]+=qt[d+7]*kf7; } }
        for(;d<head_dim;d++){ float kf=f16_bits_to_f32(kp[d]);
            #pragma unroll
            for(int t=0;t<KT;t++) dot[t]+=qsh[(long)t*head_dim+d]*kf; }
        #pragma unroll
        for(int t=0;t<KT;t++){ float sc=(p<=base_position+t)?dot[t]*scale:-3.4e38f; scores_buf[((long)t*n_heads+head)*max_pos+p]=sc; local_max[t]=fmaxf(local_max[t],sc); }
    }
    __shared__ float red[256];
    #pragma unroll 1
    for(int t=0;t<KT;t++){ red[tid]=local_max[t]; __syncthreads();
        for(int s=blockDim.x>>1;s>0;s>>=1){ if(tid<s) red[tid]=fmaxf(red[tid],red[tid+s]); __syncthreads(); }
        if(tid==0) chunkmax_buf[((long)t*n_heads+head)*n_splits+sp]=red[0]; __syncthreads(); }
}
