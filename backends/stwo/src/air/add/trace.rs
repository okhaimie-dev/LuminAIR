use std::sync::Arc;

use crate::air::tensor::{AirTensor, TensorField};
use parking_lot::Mutex;
use rayon::iter::{ParallelBridge, ParallelIterator};
use stwo_prover::core::backend::simd::m31::PackedBaseField;
use stwo_prover::core::backend::{Backend, Column, CpuBackend};
use stwo_prover::core::{
    backend::{
        simd::{m31::LOG_N_LANES, SimdBackend},
        Col,
    },
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

pub trait TensorAddTracer<F: TensorField> {
    fn generate_trace(
        log_size: u32,
        a: &AirTensor<F>,
        b: &AirTensor<F>,
    ) -> (
        ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        AirTensor<F>,
    )
    where
        Self: Backend;
}

impl TensorAddTracer<BaseField> for CpuBackend {
    fn generate_trace(
        log_size: u32,
        a: &AirTensor<BaseField>,
        b: &AirTensor<BaseField>,
    ) -> (
        ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        AirTensor<BaseField>,
    ) {
        generate_trace_cpu(log_size, a, b)
    }
}

impl TensorAddTracer<PackedBaseField> for SimdBackend {
    fn generate_trace(
        log_size: u32,
        a: &AirTensor<PackedBaseField>,
        b: &AirTensor<PackedBaseField>,
    ) -> (
        ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        AirTensor<PackedBaseField>,
    ) {
        generate_trace_simd(log_size, a, b)
    }
}

fn generate_trace_cpu(
    _log_size: u32,
    _a: &AirTensor<BaseField>,
    _b: &AirTensor<BaseField>,
) -> (
    ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
    AirTensor<BaseField>,
) {
    todo!()
}

fn generate_trace_simd(
    log_size: u32,
    a: &AirTensor<PackedBaseField>,
    b: &AirTensor<PackedBaseField>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    AirTensor<PackedBaseField>,
) {
    assert!(a.is_broadcastable_with(b), "Tensors must be broadcastable");

    // Calculate required trace size
    let max_size = a.size().max(b.size());
    assert!(log_size >= LOG_N_LANES);

    // Initialize trace columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    let trace = Arc::new(
        trace
            .into_iter()
            .map(|col| Mutex::new(col))
            .collect::<Vec<_>>(),
    );

    let c_data = Arc::new(Mutex::new(Vec::with_capacity(
        (max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES,
    )));

    // Calculate number of SIMD-packed rows needed for each tensor
    let n_rows = 1 << (log_size - LOG_N_LANES);
    let a_packed_size = (a.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;
    let b_packed_size = (b.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;

    // Fill trace with tensor data
    // Process chunks in parallel
    const CHUNK_SIZE: usize = 256;
    (0..n_rows)
        .into_iter()
        .step_by(CHUNK_SIZE)
        .par_bridge()
        .for_each(|chunk| {
            let end = (chunk + CHUNK_SIZE).min(n_rows);
            let mut local_c_data = Vec::with_capacity(CHUNK_SIZE);

            for vec_row in chunk..end {
                if vec_row < max_size {
                    let a_idx = vec_row % a_packed_size;
                    let b_idx = vec_row % b_packed_size;

                    let sum = a.data()[a_idx] + b.data()[b_idx];

                    trace[0].lock().data[vec_row] = a.data()[a_idx];
                    trace[1].lock().data[vec_row] = b.data()[b_idx];
                    trace[2].lock().data[vec_row] = sum;

                    local_c_data.push(sum);
                }
            }

            // Append local results to global c_data
            let mut c_data = c_data.lock();
            c_data.extend(local_c_data);
        });

    let trace = Arc::try_unwrap(trace)
        .unwrap()
        .into_iter()
        .map(|mutex| mutex.into_inner())
        .collect::<Vec<_>>();

    let c_data = Arc::try_unwrap(c_data).unwrap().into_inner();

    // Create output tensor C
    let c = AirTensor {
        data: c_data,
        dims: if a.size() > b.size() {
            a.dims().to_vec()
        } else {
            b.dims().to_vec()
        },
        stride: AirTensor::<PackedBaseField>::compute_stride(if a.size() > b.size() {
            a.dims()
        } else {
            b.dims()
        }),
    };

    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        c,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo_prover::core::backend::simd::m31::PackedBaseField;
    use stwo_prover::core::fields::m31::BaseField;

    fn unpack_tensor(tensor: &AirTensor<PackedBaseField>) -> Vec<u32> {
        tensor
            .data()
            .iter()
            .flat_map(|packed| packed.to_array())
            .take(tensor.size())
            .map(|x| x.0)
            .collect()
    }

    #[test]
    fn test_generate_trace_correctness() {
        let test_cases = vec![
            // Case 1: Simple 2x2 matrices
            (
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 4],
                    vec![2, 2],
                ),
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 4],
                    vec![2, 2],
                ),
                vec![2, 2],
                vec![3u32; 4], // Expected result: 1 + 2 = 3 for all elements
            ),
            // Case 2: Broadcasting scalar to matrix
            (
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(5))],
                    vec![1],
                ),
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
                    vec![2, 3],
                ),
                vec![2, 3],
                vec![6u32; 6], // Expected result: 5 + 1 = 6 for all elements
            ),
            // Case 3: Broadcasting row to matrix
            (
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 3],
                    vec![1, 3],
                ),
                AirTensor::new(
                    vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 6],
                    vec![2, 3],
                ),
                vec![2, 3],
                vec![3u32; 6], // Expected result: 1 + 2 = 3 for all elements
            ),
            // Case 4: Different values in matrices
            (
                AirTensor::create::<SimdBackend>(vec![1, 2, 3, 4], vec![2, 2]),
                AirTensor::create::<SimdBackend>(vec![5, 6, 7, 8], vec![2, 2]),
                vec![2, 2],
                vec![6, 8, 10, 12], // Element-wise addition
            ),
        ];

        for (i, (tensor_a, tensor_b, expected_dims, expected_values)) in
            test_cases.into_iter().enumerate()
        {
            // Calculate required log_size
            let max_size = tensor_a.size().max(tensor_b.size());
            let required_log_size = ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
                .next_power_of_two()
                .trailing_zeros()
                + LOG_N_LANES;

            // Generate trace and get result tensor
            let (trace, result) =
                SimdBackend::generate_trace(required_log_size, &tensor_a, &tensor_b);

            // Check trace size
            assert_eq!(
                trace.len(),
                3,
                "Case {}: Trace should have 3 columns (a, b, c)",
                i
            );
            assert_eq!(
                trace[0].len(),
                1 << required_log_size,
                "Case {}: Trace size should be 2^{}",
                i,
                required_log_size
            );

            // Verify result tensor dimensions
            assert_eq!(
                result.dims(),
                expected_dims,
                "Case {}: Result tensor has incorrect dimensions",
                i
            );

            // Unpack and verify result values
            let result_values = unpack_tensor(&result);
            assert_eq!(
                result_values, expected_values,
                "Case {}: Result tensor has incorrect values",
                i
            );

            // Verify trace values for valid entries
            let unpacked_trace_a: Vec<_> = trace[0]
                .values
                .to_cpu()
                .into_iter()
                .take(max_size)
                .collect();
            let unpacked_trace_b: Vec<_> = trace[1]
                .values
                .to_cpu()
                .into_iter()
                .take(max_size)
                .collect();
            let unpacked_trace_c: Vec<_> = trace[2]
                .values
                .to_cpu()
                .into_iter()
                .take(max_size)
                .collect();

            for j in 0..max_size {
                assert_eq!(
                    unpacked_trace_c[j],
                    unpacked_trace_a[j] + unpacked_trace_b[j],
                    "Case {}: Trace values don't satisfy a + b = c at position {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Tensors must be broadcastable")]
    fn test_generate_trace_non_broadcastable() {
        // Try to add incompatible tensors: 2x3 and 3x2
        let tensor_a = AirTensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
            vec![2, 3],
        );
        let tensor_b = AirTensor::new(
            vec![PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6],
            vec![3, 2],
        );

        let log_size = LOG_N_LANES + 1;
        SimdBackend::generate_trace(log_size, &tensor_a, &tensor_b); // Should panic
    }
}
