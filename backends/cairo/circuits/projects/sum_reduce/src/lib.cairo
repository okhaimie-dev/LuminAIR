use orion_numbers::{F64, F64Impl};
use luminair_cairo_lib::sum_reduce;

fn main(input: Span<F64>, front_size: usize, back_size: usize, dim_size: usize) -> Span<F64> {
    sum_reduce(input, front_size, back_size, dim_size)
}
