extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=src/*");
    cc::Build::new()
        .file("./src/distance.c")
        .compiler("gcc")
        .flag("-ffast-math")
        .flag("-fopt-info-vec-all")
        .flag("-fopt-info-vec-missed")
        .flag("-fdump-tree-vect-all")
        // .compiler("clang-17")
        // // .flag("-Ofast") // -O3 -ffast-math
        // .flag("-march=native")
        // .flag("-Rpass=loop-vectorize")
        // .flag("-Rpass-missed=loop-vectorize")
        // .flag("-Rpass-analysis=loop-vectorize")
        .opt_level(3)
        .debug(true)
        .compile("distance");
}
