use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::sqrt;

fn main(self: Span<F64>) -> Span<F64> {
    sqrt(self)
}
