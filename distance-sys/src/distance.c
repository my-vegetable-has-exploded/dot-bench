#include "distance.h"
#include "assert.h" 

#if TARGET_ARM

__attribute__((target("arch=armv8-a+sve")))
extern float32_t dot_f32_sve(const float32_t* a, const float32_t* b, size_t n) {
	svfloat32_t svsum = svdup_f32(0.0f);
	svbool_t pg;
	svfloat32_t sva, svb;
	for (size_t i = 0; i < n; i += svcntw()) {
		pg = svwhilelt_b32(i, n);
		sva = svld1_f32(pg, a + i);
		svb = svld1_f32(pg, b + i);
		// svab = svmul_f32_x(pg, sva, svb);
		// svsum = svadd_f32_x(pg, svsum, svab);
		svsum = svmla_f32_x(pg, svsum, sva, svb);
	}
	return svaddv_f32(svptrue_b32(), svsum);
}

__attribute__((target("arch=armv8-a+sve")))
extern float32_t dot_f32_auto_vectorization(const float32_t* __restrict a, const float32_t* __restrict b, size_t n) {
	// use __restrict to tell the compiler that the pointers a and b do not overlap and can be optimized further
	// enable fused multiply-add (FMA) instructions by floating-point control
#pragma float_control(precise, off)
	float32_t sum = 0.0;
	// see https://godbolt.org/z/64vdPva68 for more details
	for (size_t i = 0; i < n; i += 1) {
		sum += a[i] * b[i];
	}
	return sum;
}

#endif