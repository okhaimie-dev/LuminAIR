pub mod data;
pub mod fixed_point;
pub mod prim;

#[cfg(test)]
mod tests;

pub type StwoCompiler<'a> = (prim::PrimitiveCompiler,);
