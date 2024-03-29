use half::f16;

#[cfg(target_arch = "aarch64")]
#[link(name = "distance", kind = "static")]
extern "C" {
    pub fn dot_f32_sve(a: *const f32, b: *const f32, len: usize) -> f32;
    pub fn dot_f32_auto_vectorization(a: *const f32, b: *const f32, len: usize) -> f32;
    pub fn dot_i8_sve(a: *const i8, b: *const i8, len: usize) -> f32;
    pub fn dot_i8_auto_vectorization(a: *const i8, b: *const i8, len: usize) -> f32;
    pub fn dot_f16_sve(a: *const f16, b: *const f16, len: usize) -> f32;
    pub fn dot_f16_auto_vectorization(a: *const f16, b: *const f16, len: usize) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        // check sve or neon target is available
        if std::arch::is_aarch64_feature_detected!("sve") {
            println!("sve is available");
            return;
        } else if !std::arch::is_aarch64_feature_detected!("neon") {
            println!("neon is available");
            return;
        }
        println!("sve and neon are not available, skip test_aarch64");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_f32_sve() {
        if std::arch::is_aarch64_feature_detected!("sve") {
            println!("sve is available");
        } else if !std::arch::is_aarch64_feature_detected!("neon") {
            println!("sve is not available");
            return;
        }
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = unsafe { dot_f32_sve(a.as_ptr(), b.as_ptr(), a.len()) };
        assert_eq!(result, 30.0);
    }
}
