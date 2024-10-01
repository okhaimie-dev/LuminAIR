pub(crate) mod ops;

pub use ops::binary::{add, mul, rem, lt};
pub use ops::reduce::{sum_reduce, max_reduce};