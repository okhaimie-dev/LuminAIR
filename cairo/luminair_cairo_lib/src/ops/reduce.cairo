use core::num::traits::Zero;
use core::ops::AddAssign;

pub(crate) fn sum_reduce<T, +AddAssign<T, T>, +Copy<T>, +Drop<T>, +Zero<T>>(
    mut input: Span<T>, front_size: usize, back_size: usize, dim_size: usize
) -> Span<T> {
    let mut result = ArrayTrait::new();

    // Loop over front dimensions
    for i in 0
        ..front_size {
            // Loop over back dimensions
            for j in 0
                ..back_size {
                    let mut sum = Zero::zero();
                    
                    // Loop over the dimension to reduce
                    for k in 0
                        ..dim_size {
                            // Compute the original index
                            let orig_index = (i * dim_size * back_size) + (k * back_size) + j;

                            // Accumulate the sum
                            sum += *input[orig_index];
                        };

                    result.append(sum);
                }
        };

        result.span()
}
