pub mod data;
pub mod graph;
pub mod utils;

pub mod op;

#[cfg(test)]
mod tests;

pub type StwoCompiler = (op::prim::PrimitiveCompiler,);
