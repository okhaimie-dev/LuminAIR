use lazy_static::lazy_static;
use std::env;
use std::path::PathBuf;

lazy_static! {
    pub static ref COMPILED_CAIRO_PATH: PathBuf = {
        env::var("LUMINAIR_COMPILED_CAIRO_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cairo/compiled_ops")
            })
    };
}
