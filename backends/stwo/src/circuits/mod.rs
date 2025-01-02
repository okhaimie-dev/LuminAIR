use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo_prover::core::{
    backend::{
        simd::{m31::{PackedBaseField, LOG_N_LANES}, SimdBackend},
        Backend, BackendForChannel, CpuBackend,
    },
    channel::MerkleChannel,
    fields::m31::BaseField,
    pcs::PcsConfig,
    vcs::ops::MerkleHasher,
};

pub mod add;

pub trait Circuit<B: Backend> {
    type Component;
    type Proof<'a, H: MerkleHasher>;
    type Error;
    type Trace;

    /// Generates the execution trace for the circuit
    fn generate_trace(&self) -> Self::Trace;

    /// Creates proof for a given trace
    fn prove<'a, MC: MerkleChannel>(
        trace: &Self::Trace,
        config: PcsConfig,
    ) -> (Vec<Self::Component>, Self::Proof<'a, MC::H>)
    where
        B: BackendForChannel<MC>;

    /// Verifies a proof
    fn verify<'a, MC: MerkleChannel>(
        components: Vec<Self::Component>,
        proof: Self::Proof<'a, MC::H>,
        config: PcsConfig,
    ) -> Result<(), Self::Error>;
}

pub trait TensorField: Clone + Send + Sync {
    fn zero() -> Self;
}

impl TensorField for BaseField {
    fn zero() -> Self {
        BaseField::from_u32_unchecked(0)
    }
}

impl TensorField for PackedBaseField {
    fn zero() -> Self {
        PackedBaseField::broadcast(BaseField::from_u32_unchecked(0))
    }
}

#[derive(Clone, Debug)]
pub struct Tensor<F: TensorField> {
    pub data: Vec<F>,
    pub dims: Vec<usize>,
    pub stride: Vec<usize>,
}

impl<F: TensorField> Tensor<F> {
    pub fn new(data: Vec<F>, dims: Vec<usize>) -> Self {
        let stride = Self::compute_stride(&dims);
        Self { data, dims, stride }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn data(&self) -> &[F] {
        &self.data
    }

    pub fn compute_stride(dims: &[usize]) -> Vec<usize> {
        let mut stride = vec![1; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            stride[i] = stride[i + 1] * dims[i + 1];
        }
        stride
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_broadcastable_with<G: TensorField>(&self, other: &Tensor<G>) -> bool {
        let max_dims = self.dims.len().max(other.dims.len());
        let pad_self = max_dims - self.dims.len();
        let pad_other = max_dims - other.dims.len();

        (0..max_dims).all(|i| {
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
            dim_self == dim_other || dim_self == 1 || dim_other == 1
        })
    }
}

pub trait TensorPacker {
    type Field: TensorField;

    fn pack_data(data: Vec<u32>, dims: &[usize]) -> Vec<Self::Field>;
}

impl TensorPacker for CpuBackend {
    type Field = BaseField;

    fn pack_data(data: Vec<u32>, dims: &[usize]) -> Vec<Self::Field> {
        data.into_iter()
            .map(|x| BaseField::from_u32_unchecked(x % 1000))
            .collect()
    }
}

impl TensorPacker for SimdBackend {
    type Field = PackedBaseField;

    fn pack_data(data: Vec<u32>, dims: &[usize]) -> Vec<Self::Field> {
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

// Helper function to create tensors for specific backends
impl<F: TensorField> Tensor<F> {
    pub fn create<B: Backend + TensorPacker<Field = F>>(data: Vec<u32>, dims: Vec<usize>) -> Self {
        let packed_data = B::pack_data(data, &dims);
        Self::new(packed_data, dims)
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
