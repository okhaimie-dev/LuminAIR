use std::sync::Arc;

use parking_lot::Mutex;
use rayon::iter::{ParallelBridge, ParallelIterator};
use stwo_prover::core::backend::Column;
use stwo_prover::core::{
    backend::{
        simd::{
            m31::{PackedBaseField, LOG_N_LANES},
            SimdBackend,
        },
        Col,
    },
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

use crate::ops::mul::SCALE_FACTOR_INV;
use crate::tensor::AirTensor;

pub(super) fn generate_trace<'a>(
    log_size: u32,
    a: &'a AirTensor<'a, PackedBaseField>,
    b: &'a AirTensor<'a, PackedBaseField>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    AirTensor<'static, PackedBaseField>,
) {
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

                    // Get values from tensors
                    let a_val = a.data()[a_idx];
                    let b_val = b.data()[b_idx];

                    // Fixed point multiplication:
                    // To get the correct scaling, we need to:
                    // 1. Multiply a * b
                    // 2. Divide by scale factor (2^DEFAULT_SCALE)
                    let scaled_mul = {
                        let ab = a_val * b_val;
                        println!("a_val: {:?}", a_val);
                        println!("b_val: {:?}", b_val);
                        println!("ab: {:?}", ab);

                        // Shift the underlying SIMD vector and create new PackedM31
                        // unsafe {
                        //     PackedM31::from_simd_unchecked(ab.into_simd() >> DEFAULT_SCALE as u32)
                        // }
                        ab * PackedBaseField::broadcast(*SCALE_FACTOR_INV)
                    };

                    println!("scaled_mul: {:?}", scaled_mul);

                    // Store values in trace
                    trace[0].lock().data[vec_row] = a_val;
                    trace[1].lock().data[vec_row] = b_val;
                    trace[2].lock().data[vec_row] = scaled_mul;

                    local_c_data.push(scaled_mul);
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
    let c = AirTensor::Owned {
        data: c_data,
        dims: if a.size() > b.size() {
            a.dims().to_vec()
        } else {
            b.dims().to_vec()
        },
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
    use numerair::fixed_points::{pack_floats, unpack_floats, DEFAULT_SCALE};

    #[test]
    fn test_generate_trace_correctness() {
        let binding_a_1 = pack_floats(&[1.0; 4], DEFAULT_SCALE, 0.0);
        let binding_b_1 = pack_floats(&[2.0; 4], DEFAULT_SCALE, 0.0);
        let binding_a_2 = pack_floats(&[5.0; 1], DEFAULT_SCALE, 0.0);
        let binding_b_2 = pack_floats(&[1.0; 6], DEFAULT_SCALE, 0.0);
        let binding_a_3 = pack_floats(&[1.0; 3], DEFAULT_SCALE, 0.0);
        let binding_b_3 = pack_floats(&[2.0; 6], DEFAULT_SCALE, 0.0);

        let test_cases = vec![
            // Case 1: Simple 2x2 matrices
            (
                AirTensor::new(&binding_a_1, vec![2, 2]),
                AirTensor::new(&binding_b_1, vec![2, 2]),
                vec![2, 2],
                pack_floats(&[2.0; 4], DEFAULT_SCALE, 0.0), // Expected result: 1 * 2 = 2 for all elements
            ),
            // Case 2: Broadcasting scalar to matrix
            (
                AirTensor::new(&binding_a_2, vec![1]),
                AirTensor::new(&binding_b_2, vec![2, 3]),
                vec![2, 3],
                pack_floats(&[5.0; 6], DEFAULT_SCALE, 0.0), // Expected result: 5 * 1 = 5 for all elements
            ),
            // Case 3: Broadcasting row to matrix
            (
                AirTensor::new(&binding_a_3, vec![1, 3]),
                AirTensor::new(&binding_b_3, vec![2, 3]),
                vec![2, 3],
                pack_floats(&[2.0; 6], DEFAULT_SCALE, 0.0), // Expected result: 1 * 2 = 2 for all elements
            ),
            // Case 4: Different values in matrices
            (
                AirTensor::create::<SimdBackend>(vec![1, 2, 3, 4], vec![2, 2]),
                AirTensor::create::<SimdBackend>(vec![5, 6, 7, 8], vec![2, 2]),
                vec![2, 2],
                pack_floats(&[5.0, 12.0, 21.0, 32.0], DEFAULT_SCALE, 0.0), // Element-wise multiplication
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
            let (trace, result) = generate_trace(required_log_size, &tensor_a, &tensor_b);

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

            let result_values = unpack_floats(result.data(), DEFAULT_SCALE, 0.0, result.size());
            let expected_values = unpack_floats(
                &expected_values,
                DEFAULT_SCALE,
                0.0,
                expected_dims.iter().product(),
            );

            assert_eq!(
                result_values, expected_values,
                "Case {:?}: Result tensor has incorrect values",
                i
            );
        }
    }
}
