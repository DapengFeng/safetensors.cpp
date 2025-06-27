fn main() -> miette::Result<()> {
    cxx_build::bridge("src/lib.rs").std("c++20");

    println!("cargo:rerun-if-changed=src/lib.rs");
    Ok(())
}
