use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo_prover::core::{
    backend::simd::m31::{PackedBaseField, LOG_N_LANES},
    fields::m31::BaseField,
};

pub mod add;

pub trait Circuit {
    fn trace_generator(/* ... */) {/* ... */}

    fn prover(/* ... */) {/* ... */}

    fn verifier(/* ... */) {/* ... */}
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<PackedBaseField>,
    pub dims: Vec<usize>,
    pub stride: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<PackedBaseField>, dims: Vec<usize>) -> Self {
        let stride = Self::compute_stride(&dims);
        Self { data, dims, stride }
    }

    pub fn compute_stride(dims: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        stride
    }

    // Get total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    // Check if tensors are broadcastable
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let max_dims = self.dims.len().max(other.dims.len());
        let pad_self = max_dims - self.dims.len();
        let pad_other = max_dims - other.dims.len();

        for i in 0..max_dims {
            let dim_self = if i < pad_self {
                1
            } else {
                self.dims[i - pad_self]
            };
            let dim_other = if i < pad_other {
                1
            } else {
                other.dims[i - pad_other]
            };

            if dim_self != dim_other && dim_self != 1 && dim_other != 1 {
                return false;
            }
        }
        true
    }

    // helper function to create SIMD-efficient packed data
    pub fn pack_data(data: Vec<u32>, dims: &[usize]) -> Vec<PackedBaseField> {
        let total_size = dims.iter().product::<usize>();
        let n_packed = (total_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;

        (0..n_packed)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx << LOG_N_LANES;
                let mut lane_values = [0u32; 1 << LOG_N_LANES];

                for (i, lane) in lane_values.iter_mut().enumerate() {
                    let data_idx = start + i;
                    *lane = if data_idx < data.len() {
                        data[data_idx] % 1000
                    } else {
                        0
                    };
                }

                PackedBaseField::from_array(lane_values.map(|x| BaseField::from_u32_unchecked(x)))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_broadcasting() {
        let a = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 2],
            vec![2, 1],
        );
        let b = Tensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 3],
            vec![1, 3],
        );
        assert!(a.is_broadcastable_with(&b));
    }
}
