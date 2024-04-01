#include "distance.h"
#include "assert.h" 

#if TARGET_ARM

__attribute__((target("arch=armv8-a+sve")))
extern float dot_f32_sve(const float* a, const float* b, size_t n) {
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
extern float dot_f32_auto_vectorization(const float* __restrict a, const float* __restrict b, size_t n) {
	// use __restrict to tell the compiler that the pointers a and b do not overlap and can be optimized further
	// enable fused multiply-add (FMA) instructions by floating-point control
#pragma float_control(precise, off)
	float sum = 0.0;
	// see https://godbolt.org/z/64vdPva68 for more details
	for (size_t i = 0; i < n; i += 1) {
		sum += a[i] * b[i];
	}
	return sum;
}

__attribute__((target("arch=armv8-a+sve")))
extern float dot_i8_auto_vectorization(const int8_t* __restrict a, const int8_t* __restrict b, size_t n) {
	// use __restrict to tell the compiler that the pointers a and b do not overlap and can be optimized further
	// it won't use svdot when compiling using clang
	// see https://godbolt.org/z/Tx4vzYfhs for more details
	int32_t sum = 0;
	for (size_t i = 0; i < n; i += 1) {
		sum += a[i] * b[i];
	}
	return sum;
}

__attribute__((target("arch=armv8-a+sve")))
extern float dot_i8_sve(const int8_t* __restrict a, const int8_t* __restrict b, size_t n) {
	// use __restrict to tell the compiler that the pointers a and b do not overlap and can be optimized further
	svint32_t svsum = svdup_s32(0);
	svbool_t pg;
	svint8_t sva, svb;
	for (size_t i = 0; i < n; i += svcntb()) {
		pg = svwhilelt_b8(i, n);
		sva = svld1_s8(pg, a + i);
		svb = svld1_s8(pg, b + i);
		// clang will not use svdot automatically
		// see https://godbolt.org/z/Tx4vzYfhs for more details
		svsum = svdot_s32(svsum, sva, svb);
	}
	return svaddv_s32(svptrue_b32(), svsum);
}

__attribute__((target("arch=armv8-a+sve+fp16")))
extern float dot_f16_sve(const _Float16* __restrict a, const _Float16* __restrict b, size_t n) {
	svfloat16_t svsum = svdup_f16(0.0f);
	svbool_t pg;
	svfloat16_t sva, svb;
	for (size_t i = 0; i < n; i += svcnth()) {
		pg = svwhilelt_b16(i, n);
		sva = svld1_f16(pg, a + i);
		svb = svld1_f16(pg, b + i);
		svsum = svmla_f16_x(pg, svsum, sva, svb);
	}
	return svaddv_f16(svptrue_b16(), svsum);
}

__attribute__((target("arch=armv8-a+sve+fp16")))
extern float dot_f16_auto_vectorization(const _Float16* __restrict a, const _Float16* __restrict b, size_t n) {
	// use __restrict to tell the compiler that the pointers a and b do not overlap and can be optimized further
	// enable fused multiply-add (FMA) instructions by floating-point control
#pragma float_control(precise, off)
	_Float16 sum = 0.0;
	// it would be auto-vectorized by the compiler, see https://godbolt.org/z/cabsh6PPG for more details
	for (size_t i = 0; i < n; i += 1) {
		sum += a[i] * b[i];
	}
	return  (float)sum;
}

#endif