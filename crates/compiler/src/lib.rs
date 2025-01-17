use std::path::PathBuf;

use luminal::prelude::GenericCompiler;
use prim::PrimitiveCompiler;

pub mod utils;
pub mod data;
pub mod prim;

#[cfg(test)]
mod tests;

pub type StwoCompiler = (prim::PrimitiveCompiler,);

pub fn init_compiler(trace_registry: Option<PathBuf>) -> (GenericCompiler, StwoCompiler) {
    let config = prim::Config { trace_registry };
    let primitive_compiler = PrimitiveCompiler::new(config);
    (GenericCompiler::default(), (primitive_compiler,))
}
