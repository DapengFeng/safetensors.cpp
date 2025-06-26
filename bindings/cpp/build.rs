fn main() -> miette::Result<()> {
    cxx_build::bridge("src/lib.rs").std("c++17");

    println!("cargo:rerun-if-changed=src/lib.rs");
    Ok(())
}
