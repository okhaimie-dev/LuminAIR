use std::env;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("compiled_ops");
    fs::create_dir_all(&dest_path).unwrap();

    let src_path = Path::new("circuits/compiled_ops");
    for entry in WalkDir::new(src_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
            let dest_file = dest_path.join(path.file_name().unwrap());
            fs::copy(path, dest_file).unwrap();
        }
    }

    println!("cargo:rerun-if-changed=circuits/compiled_ops");
}
