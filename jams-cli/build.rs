fn main() {
    println!("cargo:rustc-link-arg-bin=jams-cli=-Wl,-rpath,/usr/local/lib");
}