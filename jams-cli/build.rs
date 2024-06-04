use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    if target_os == "macos" {
        println!("cargo:rustc-link-arg-bin=jams-cli=-Wl,-rpath,@loader_path/../lib");
        println!("cargo:rustc-link-arg-bin=jams-cli=-Wl,-rpath,/usr/local/lib");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-arg-bin=jams-cli=-Wl,-rpath,$ORIGIN/../lib");
        println!("cargo:rustc-link-arg-bin=jams-cli=-Wl,-rpath,/usr/local/lib");
    }

    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
