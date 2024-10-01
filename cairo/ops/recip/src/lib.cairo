use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::recip;

fn main(self: Span<F64>) -> Span<F64> {
    recip(self)
}
