// SIROCCO Phase P — M0a TRAFFIC-REALITY kill gate.
// Lifts attention_batched's read pattern (uint4 K-dot #443 + scalar weighted-V) VERBATIM: grid =
// (k_tokens * n_heads) blocks, one block per (query-token t, head), each re-streaming the full
// prefix K+V for its kv_head. Times k=8 vs k=1 at the SAME base. If the k-token re-reads miss to
// DRAM (bandwidth-bound), t(k=8)/t(k=1) ~ 8; if L2 absorbs them, ~1. k=1 underfills the 30 SMs so
// the ratio is biased toward KILL -> a high ratio is a STRONG go. GO >=5, KILL <=2.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__device__ __forceinline__ float f16b(unsigned short h){ return __half2float(__ushort_as_half(h)); }

extern "C" __global__ void attn_bench(
    const float* __restrict__ q, const unsigned short* __restrict__ cache_k,
    const unsigned short* __restrict__ cache_v, float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int base_position, int max_pos, float scale,
    int q_per_token, int k_tokens)
{
    int t = blockIdx.x / n_heads, head = blockIdx.x % n_heads;
    if (t >= k_tokens) return;
    int position_count = base_position + t + 1;
    int repeats = n_heads / n_kv_heads, kv_head = head / repeats;
    const float* qh = q + (long)t * q_per_token + (long)head * head_dim;
    const unsigned short* kbase = cache_k + (long)kv_head * max_pos * head_dim;
    const unsigned short* vbase = cache_v + (long)kv_head * max_pos * head_dim;
    extern __shared__ float shared[];
    float* qsh = shared; float* scores = shared + head_dim;
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();
    int kd8 = ((head_dim & 7) == 0) ? head_dim : 0;
    for (int p = tid; p < position_count; p += blockDim.x) {         // K-read (uint4, #443)
        const unsigned short* kp = kbase + (long)p * head_dim;
        float dot = 0.0f; int d = 0;
        for (; d < kd8; d += 8) {
            uint4 kv = *reinterpret_cast<const uint4*>(kp + d);
            const unsigned short* k8 = reinterpret_cast<const unsigned short*>(&kv);
            dot += qsh[d+0]*f16b(k8[0]); dot += qsh[d+1]*f16b(k8[1]); dot += qsh[d+2]*f16b(k8[2]); dot += qsh[d+3]*f16b(k8[3]);
            dot += qsh[d+4]*f16b(k8[4]); dot += qsh[d+5]*f16b(k8[5]); dot += qsh[d+6]*f16b(k8[6]); dot += qsh[d+7]*f16b(k8[7]);
        }
        for (; d < head_dim; d++) dot += qsh[d]*f16b(kp[d]);
        scores[p] = dot * scale;
    }
    __syncthreads();
    // weighted-V (scalar, head_dim threads active — as in attention_batched)
    for (int did = tid; did < head_dim; did += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < position_count; p++) acc += scores[p] * f16b(vbase[(long)p*head_dim+did]);
        out[(long)t*q_per_token + (long)head*head_dim + did] = acc;
    }
}

double run(int k, int base, int n_heads, int n_kv, int hd, int iters){
    int max_pos = base + 16; long qn=(long)k*n_heads*hd;
    float *q,*out; unsigned short *ck,*cv;
    CK(cudaMalloc(&q,qn*4)); CK(cudaMemset(q,1,qn*4)); CK(cudaMalloc(&out,qn*4));
    CK(cudaMalloc(&ck,(long)n_kv*max_pos*hd*2)); CK(cudaMemset(ck,1,(long)n_kv*max_pos*hd*2));
    CK(cudaMalloc(&cv,(long)n_kv*max_pos*hd*2)); CK(cudaMemset(cv,1,(long)n_kv*max_pos*hd*2));
    dim3 grid(k*n_heads,1,1); int block=128; size_t sh=(hd+base+16)*4; float scale=0.125f;
    for(int i=0;i<5;i++) attn_bench<<<grid,block,sh>>>(q,ck,cv,out,n_heads,n_kv,hd,base,max_pos,scale,n_heads*hd,k);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); std::vector<double> ts;
    for(int i=0;i<iters;i++){ CK(cudaEventRecord(a));
        attn_bench<<<grid,block,sh>>>(q,ck,cv,out,n_heads,n_kv,hd,base,max_pos,scale,n_heads*hd,k);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms,a,b)); ts.push_back(ms); }
    std::sort(ts.begin(),ts.end());
    cudaFree(q);cudaFree(out);cudaFree(ck);cudaFree(cv);
    return ts[ts.size()/2];
}

int main(int argc,char**argv){
    int base = argc>1?atoi(argv[1]):8000;
    struct Cfg{const char*name;int nh,nkv,hd;};
    Cfg cfgs[] = {{"Llama-3.2-1B (32/8, hd64)",32,8,64},{"Llama-3.2-3B (24/8, hd128)",24,8,128}};
    printf("M0a TRAFFIC-REALITY @ base=%d  (peak ~271 GB/s; single-read min bytes = n_kv*base*hd*2*2)\n",base);
    for(auto c:cfgs){
        double t1=run(1,base,c.nh,c.nkv,c.hd,60), t8=run(8,base,c.nh,c.nkv,c.hd,60);
        double minB=(double)c.nkv*base*c.hd*2*2/1e9; // GB, single-read K+V
        double naiveB8=(double)8*c.nh*base*c.hd*2*2/1e9; // naive: every (t,head) block reads full K+V
        double bw8=naiveB8/(t8/1e3);
        printf("  %-28s t(k=1)=%.3fms t(k=8)=%.3fms  RATIO=%.2f  | k=8 naive-traffic BW=%.0f GB/s (min-read %.1f MB, naive %.0f MB)\n",
            c.name,t1,t8,t8/t1,bw8, minB*1e3, naiveB8*1e3);
    }
    printf("VERDICT: RATIO>=5 => re-reads miss to DRAM, flash win REAL (GO). RATIO<=2 => L2 absorbs, KILL.\n");
    return 0;
}
