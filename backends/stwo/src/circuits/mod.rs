use rayon::iter::{IntoParallelIterator, ParallelIterator};
use stwo_prover::core::{
    backend::{
        simd::{
            m31::{PackedBaseField, LOG_N_LANES},
            SimdBackend,
        },
        BackendForChannel,
    },
    channel::MerkleChannel,
    fields::{m31::BaseField, IntoSlice},
    pcs::PcsConfig,
    vcs::{
        blake2_hash::{Blake2sHash, Blake2sHasher},
        ops::MerkleHasher,
    },
};

pub mod add;

pub trait Circuit {
    type PublicInputs;
    type Component;
    type Proof<'a, H: MerkleHasher>;
    type Error;
    type Trace;

    /// Generates the execution trace for the circuit
    fn generate_trace(&self) -> (Self::Trace, Self::PublicInputs);

    /// Creates proof for a given trace
    fn prove<'a, MC: MerkleChannel>(
        trace: &Self::Trace,
        public_inputs: &'a Self::PublicInputs,
        config: PcsConfig,
    ) -> (Vec<Self::Component>, Self::Proof<'a, MC::H>)
    where
        SimdBackend: BackendForChannel<MC>;

    /// Verifies a proof
    fn verify<'a, MC: MerkleChannel>(
        components: Vec<Self::Component>,
        proof: Self::Proof<'a, MC::H>,
        config: PcsConfig,
    ) -> Result<(), Self::Error>;
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

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn data(&self) -> &[PackedBaseField] {
        &self.data
    }

    pub fn hash(&self) -> Blake2sHash {
        let mut hasher = Blake2sHasher::new();

        // Hash dimensions
        hasher.update(&(self.dims.len() as u64).to_le_bytes());
        for dim in &self.dims {
            hasher.update(&(*dim as u64).to_le_bytes());
        }

        // Unpack SIMD data into base field values
        let unpacked_data: Vec<BaseField> = self
            .data
            .iter()
            .flat_map(|packed| packed.to_array())
            .collect();

        // Hash unpacked data
        hasher.update(IntoSlice::<u8>::into_slice(&unpacked_data));
        hasher.finalize()
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
    pub fn is_broadcastable_with(&self, other: &Self) -> bool {
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
