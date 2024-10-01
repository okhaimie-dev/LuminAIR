use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::log2;

fn main(self: Span<F64>) -> Span<F64> {
    log2(self)
}
