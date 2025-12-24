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
  asm volatile("mov.u32 %0, %1;" : "+r"(r3) : "r"(blockIdx.x) : );
  asm volatile("mov.u32 %0, %1;" : "+r"(r4) : "r"(blockDim.x) : );
  asm volatile("mov.u32 %0, %1;" : "+r"(r5) : "r"(threadIdx.x) : );
  asm volatile("mad.lo.s32 %0, %1, %2, %3;" : "+r"(r1) : "r"(r3), "r"(r4), "r"(r5) : );
  asm volatile("{ .reg .pred %ptmp0; setp.ne.u32 %ptmp0, %0, 0; setp.ge.s32 %ptmp0, %1, %2; selp.u32 %0, 1, 0, %ptmp0; }" : "+r"(p1) : "r"(r1), "r"(r2) : );
  if (p1 != 0) goto L__BB0_2;
  asm volatile("cvta.to.global.u64 %0, %1;" : "+l"(rd3) : "l"(rd2) : );
  asm volatile("cvta.to.global.u64 %0, %1;" : "+l"(rd4) : "l"(rd1) : );
  asm volatile("mul.wide.s32 %0, %1, 4;" : "+l"(rd5) : "r"(r1) : );
  asm volatile("add.s64 %0, %1, %2;" : "+l"(rd6) : "l"(rd4), "l"(rd5) : );
  asm volatile("add.s64 %0, %1, %2;" : "+l"(rd7) : "l"(rd3), "l"(rd5) : );
  asm volatile("ld.global.f32 %0, [%1];" : "+f"(f2) : "l"(rd7) : "memory");
  asm volatile("ld.global.f32 %0, [%1];" : "+f"(f3) : "l"(rd6) : "memory");
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "+f"(f4) : "f"(f3), "f"(f1), "f"(f2) : );
  asm volatile("st.global.f32 [%0], %1;" :  : "l"(rd7), "f"(f4) : "memory");
L__BB0_2:
  return;
}
