pub(crate) mod ops;

#[derive(Drop, Copy)]
pub struct Tensor<T> {
    pub data: Span<T>,
}
