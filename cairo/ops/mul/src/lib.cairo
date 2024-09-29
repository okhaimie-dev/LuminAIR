use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::mul;

fn main(x: Span<F64>, y: Span<F64>) -> Span<F64> {
    mul(x, y)
}
