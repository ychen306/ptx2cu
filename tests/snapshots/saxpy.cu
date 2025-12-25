#include <cuda_fp16.h>

extern "C" __global__ void _Z5saxpyifPKfPf(unsigned int _Z5saxpyifPKfPf_param_0, float _Z5saxpyifPKfPf_param_1, unsigned long long _Z5saxpyifPKfPf_param_2, unsigned long long _Z5saxpyifPKfPf_param_3)
{
  unsigned int p0, p1, r0, r1, r2, r3, r4, r5;
  float f0, f1, f2, f3, f4;
  unsigned long long rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7;
  r2 = reinterpret_cast<unsigned int*>(&_Z5saxpyifPKfPf_param_0)[0];
  f1 = reinterpret_cast<float*>(&_Z5saxpyifPKfPf_param_1)[0];
  rd1 = reinterpret_cast<unsigned long long*>(&_Z5saxpyifPKfPf_param_2)[0];
  rd2 = reinterpret_cast<unsigned long long*>(&_Z5saxpyifPKfPf_param_3)[0];
  r3 = blockIdx.x;
  r4 = blockDim.x;
  r5 = threadIdx.x;
  r1 = (unsigned int)(((int)(((long long)((int)(r3)) * (long long)((int)(r4)))) + (int)(r5)));
  p1 = ((int)(r1) >= (int)(r2));
  if (p1 != 0) goto L__BB0_2;
  asm volatile("cvta.to.global.u64 %0, %1;" : "+l"(rd3) : "l"(rd2) : );
  asm volatile("cvta.to.global.u64 %0, %1;" : "+l"(rd4) : "l"(rd1) : );
  rd5 = (unsigned long long)(((long long)((int)(r1)) * 4));
  rd6 = (rd4 + rd5);
  rd7 = (rd3 + rd5);
  f2 = reinterpret_cast<float*>(rd7)[0];
  f3 = reinterpret_cast<float*>(rd6)[0];
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "+f"(f4) : "f"(f3), "f"(f1), "f"(f2) : );
  reinterpret_cast<float*>(rd7)[0] = f4;
L__BB0_2:
  return;
}
