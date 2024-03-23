# Benchmark

- install rust
- change to nightly
```bash
rustup override set nightly
```
- run benchmark
```bash
cargo bench
```

## my result

```bash
ubuntu@ip-172-31-65-183:~/rust/dot-bench$ cargo bench
    Finished `bench` profile [optimized] target(s) in 0.01s
     Running unittests src/lib.rs (target/release/deps/dot_bench-787c47d968a41971)

running 6 tests
test tests::test_dot ... ignored
test tests::bench_dot_avx512     ... bench:         321 ns/iter (+/- 9)
test tests::bench_dot_f32        ... bench:       6,079 ns/iter (+/- 26)
test tests::bench_dot_fallback   ... bench:         430 ns/iter (+/- 8)
test tests::bench_dot_vnni       ... bench:         229 ns/iter (+/- 3)
test tests::bench_dot_vnni_fault ... bench:         185 ns/iter (+/- 3)

test result: ok. 0 passed; 0 failed; 1 ignored; 5 measured; 0 filtered out; finished in 4.94s
```
