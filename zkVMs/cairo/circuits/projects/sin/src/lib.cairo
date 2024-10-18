use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::sin;

fn main(self: Span<F64>) -> Span<F64> {
    sin(self)
}
