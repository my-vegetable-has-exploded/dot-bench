#![feature(test)]

extern crate test;

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon",
    "aarch64+sve"
))]
pub fn dot_i8_fallback(x: &[i8], y: &[i8]) -> f32 {
    // i8 * i8 fall in range of i16. Since our length is less than (2^16 - 1), the result won't overflow.
    let mut sum = 0;
    assert_eq!(x.len(), y.len());
    let length = x.len();
    // according to https://godbolt.org/z/ff48vW4es, this loop will be autovectorized
    for i in 0..length {
        sum += (x[i] as i16 * y[i] as i16) as i32;
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,bmi2")]
pub unsafe fn dot_i8_avx512vnni(x: &[i8], y: &[i8]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), y.len());
    let mut sum = 0;
    let mut i = x.len();
    let mut p_x = x.as_ptr() as *const i8;
    let mut p_y = y.as_ptr() as *const i8;
    let mut vec_x;
    let mut vec_y;
    unsafe {
        let mut result = _mm512_setzero_si512();
        let zero = _mm512_setzero_si512();
        while i > 0 {
            if i < 64 {
                let mask = _bzhi_u64(0xFFFF_FFFF_FFFF_FFFF, i as u32);
                vec_x = _mm512_maskz_loadu_epi8(mask, p_x);
                vec_y = _mm512_maskz_loadu_epi8(mask, p_y);
                i = 0;
            } else {
                vec_x = _mm512_loadu_epi8(p_x);
                vec_y = _mm512_loadu_epi8(p_y);
                i -= 64;
                p_x = p_x.add(64);
                p_y = p_y.add(64);
            }
            // there are only _mm512_dpbusd_epi32 support, dpbusd will zeroextend a[i] and signextend b[i] first, so we need to convert a[i] positive and change corresponding b[i] to get right result.
            let neg_mask = _mm512_movepi8_mask(vec_x);
            vec_x = _mm512_mask_abs_epi8(vec_x, neg_mask, vec_x);
            vec_y = _mm512_mask_sub_epi8(vec_y, neg_mask, zero, vec_y);
            result = _mm512_dpbusd_epi32(result, vec_x, vec_y);
        }
        sum += _mm512_reduce_add_epi32(result);
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,bmi2")]
pub unsafe fn dot_i8_avx512vnni_fault(x: &[i8], y: &[i8]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), y.len());
    let mut sum = 0;
    let mut i = x.len();
    let mut p_x = x.as_ptr() as *const i8;
    let mut p_y = y.as_ptr() as *const i8;
    let mut vec_x;
    let mut vec_y;
    unsafe {
        let mut result = _mm512_setzero_si512();
        while i > 0 {
            if i < 64 {
                let mask = _bzhi_u64(0xFFFF_FFFF_FFFF_FFFF, i as u32);
                vec_x = _mm512_maskz_loadu_epi8(mask, p_x);
                vec_y = _mm512_maskz_loadu_epi8(mask, p_y);
                i = 0;
            } else {
                vec_x = _mm512_loadu_epi8(p_x);
                vec_y = _mm512_loadu_epi8(p_y);
                i -= 64;
                p_x = p_x.add(64);
                p_y = p_y.add(64);
            }
            result = _mm512_dpbusd_epi32(result, vec_x, vec_y);
        }
        sum += _mm512_reduce_add_epi32(result);
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,bmi2")]
/// for llvm auto vectorization, axv512 is disabled default due to the frequency drops in some cpu architecture (although this problem is much better nowadays).
pub unsafe fn dot_i8_avx512(x: &[i8], y: &[i8]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(x.len(), y.len());
    let mut sum = 0;
    let mut i = x.len();
    let mut p_x = x.as_ptr() as *const i8;
    let mut p_y = y.as_ptr() as *const i8;
    let mut vec_x;
    let mut vec_y;
    unsafe {
        let mut result = _mm512_setzero_si512();
        while i > 0 {
            if i < 32 {
                let mask = _bzhi_u32(0xFFFF_FFFF, i as u32);
                vec_x = _mm512_maskz_cvtepi8_epi16(mask, _mm256_loadu_si256(p_x as *const __m256i));
                vec_y = _mm512_maskz_cvtepi8_epi16(mask, _mm256_loadu_si256(p_y as *const __m256i));
                i = 0;
            } else {
                vec_x = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p_x as *const __m256i));
                vec_y = _mm512_cvtepi8_epi16(_mm256_loadu_si256(p_y as *const __m256i));
                i -= 32;
                p_x = p_x.add(32);
                p_y = p_y.add(32);
            }

            result = _mm512_add_epi32(result, _mm512_madd_epi16(vec_x, vec_y));
        }
        sum += _mm512_reduce_add_epi32(result);
    }
    sum as f32
}

// #[cfg(target_arch = "x86_64")]
// rust don't support AVX_VNNI_INT8 yet.
// unsafe fn dot_i8_avxvnni(x: &[i8], y: &[i8]) -> f32 {
// }

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn i8_quantization(vector: Vec<f32>) -> (Vec<i8>, f32, f32) {
    let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // we use 254.0 to avoid overflow.
    let alpha = (max - min) / 254.0;
    let offset = (max + min) / 2.0;
    let result = vector
        .iter()
        .map(|&x| ((x - offset) / alpha) as i8)
        .collect();
    (result, alpha, offset)
}

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn i8_dequantization(vector: &[i8], alpha: f32, offset: f32) -> Vec<f32> {
    vector
        .iter()
        .map(|&x| (x as f32 * alpha + offset))
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use test::Bencher;

    fn new_random_vec_i8(size: usize) -> Vec<i8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // we use -127..127 to avoid overflow, since (0-(-128)) would overflow range of i8
        (0..size).map(|_| rng.gen_range(-127..127)).collect()
    }

    fn new_random_vec_f32(size: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(-127.0..128.0)).collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot() {
        // check vnni target is available
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("bmi2"))
        {
            println!("avx512vnni is not available, skip test_dot");
            return;
        }
        let length = 10000;
        let x = new_random_vec_i8(length);
        let y = new_random_vec_i8(length);
        // let x = vec![-1, -2];
        // let y = vec![1, 2];
        // println!("x: {:?}", x);
        // println!("y: {:?}", y);
        assert_eq!(
            unsafe { dot_i8_avx512vnni(&x, &y) },
            dot_i8_fallback(&x, &y)
        );
        assert_eq!(unsafe { dot_i8_avx512(&x, &y) }, dot_i8_fallback(&x, &y));
    }

    #[bench]
    #[cfg(target_arch = "x86_64")]
    fn bench_dot_vnni_fault(b: &mut Bencher) {
        // check vnni target is available
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("bmi2"))
        {
            println!("avx512vnni is not available, skip bench_dot_vnni_fault");
            return;
        }
        let x = new_random_vec_i8(10000);
        let y = new_random_vec_i8(10000);
        b.iter(|| unsafe { dot_i8_avx512vnni_fault(&x, &y) });
    }

    #[bench]
    #[cfg(target_arch = "x86_64")]
    fn bench_dot_vnni(b: &mut Bencher) {
        // check vnni target is available
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("bmi2"))
        {
            println!("avx512vnni is not available, skip bench_dot_vnni");
            return;
        }
        let x = new_random_vec_i8(10000);
        let y = new_random_vec_i8(10000);
        b.iter(|| unsafe { dot_i8_avx512vnni(&x, &y) });
    }

    #[bench]
    #[cfg(target_arch = "x86_64")]
    fn bench_dot_avx512(b: &mut Bencher) {
        // check avx512 target is available
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("bmi2"))
        {
            println!("avx512 is not available, skip bench_dot_avx512");
            return;
        }
        let x = new_random_vec_i8(10000);
        let y = new_random_vec_i8(10000);
        b.iter(|| unsafe { dot_i8_avx512(&x, &y) });
    }

    #[bench]
    fn bench_dot_i8_fallback(b: &mut Bencher) {
        let x = new_random_vec_i8(10000);
        let y = new_random_vec_i8(10000);
        b.iter(|| dot_i8_fallback(&x, &y));
    }

    #[bench]
    fn bench_dot_f32(b: &mut Bencher) {
        let x = new_random_vec_f32(10000);
        let y = new_random_vec_f32(10000);
        b.iter(|| dot_f32(&x, &y));
    }
}
