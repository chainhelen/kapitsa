fn main() {
    println!("cargo:rerun-if-env-changed=KAPITSA_LIB_DIR");
    let dir =
        std::env::var("KAPITSA_LIB_DIR").expect("set KAPITSA_LIB_DIR to the CMake build directory");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=kapitsa_sdk");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
