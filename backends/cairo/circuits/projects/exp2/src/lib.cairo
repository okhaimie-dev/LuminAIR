use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::exp2;

fn main(self: Span<F64>) -> Span<F64> {
    exp2(self)
}
