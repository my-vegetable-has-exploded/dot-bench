#include <stddef.h>
#include <stdint.h>

// Compiling for Arm
#if !defined(TARGET_ARM)
#if defined(__aarch64__) || defined(_M_ARM64)
#define TARGET_ARM 1
#else
#define TARGET_ARM 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(TARGET_ARM)

#if TARGET_ARM
#include <arm_sve.h>
// #include "/usr/include/clang/17/include/arm_sve.h"
#endif

#if TARGET_ARM

extern float dot_f32_sve(const float* a, const float* b, size_t n);

extern float dot_f32_auto_vectorization(const float* __restrict a, const float* __restrict b, size_t n);

extern float dot_i8_sve(const int8_t* __restrict a, const int8_t* __restrict b, size_t n);

extern float dot_i8_auto_vectorization(const int8_t* __restrict a, const int8_t* __restrict b, size_t n);

extern float dot_f16_sve(const _Float16* __restrict a, const _Float16* __restrict b, size_t n);

extern float dot_f16_auto_vectorization(const _Float16* __restrict a, const _Float16* __restrict b, size_t n);

#endif