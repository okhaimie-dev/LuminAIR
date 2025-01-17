use crate::tensor::AirTensor;
use num_traits::identities::Zero;
use numerair::FixedPoint;
use stwo_prover::core::backend::Column;
use stwo_prover::core::{
    backend::{
        simd::{m31::PackedBaseField, SimdBackend},
        Col,
    },
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

pub(super) fn generate_trace<'a>(
    log_size: u32,
    a: &'a AirTensor<'a, PackedBaseField>,
    b: &'a AirTensor<'a, PackedBaseField>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    AirTensor<'static, PackedBaseField>,
) {
    // Calculate trace size and initialize columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    // Calculate actual size needed
    let size = a.size().max(b.size());

    // Prepare output data
    let mut c_data = Vec::with_capacity(size);

    // Fill trace and generate output data
    for i in 0..trace_size {
        if i < size {
            // Get values with broadcasting
            let a_val = a.data()[i % a.data().len()];
            let b_val = b.data()[i % b.data().len()];
            let sum = a_val.fixed_add(b_val);

            trace[0].set(i, a_val.to_array()[0]);
            trace[1].set(i, b_val.to_array()[0]);
            trace[2].set(i, sum.to_array()[0]);

            if i < size {
                c_data.push(sum);
            }
        } else {
            // Pad remaining trace with zeros
            trace[0].set(i, BaseField::zero());
            trace[1].set(i, BaseField::zero());
            trace[2].set(i, BaseField::zero());
        }
    }

    // Create output tensor
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
    use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
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
        let binding_a_1 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 4];
        let binding_b_1 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 4];
        let binding_a_2 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(5))];
        let binding_b_2 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 6];
        let binding_a_3 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(1)); 3];
        let binding_b_3 = [PackedBaseField::broadcast(BaseField::from_u32_unchecked(2)); 6];
        let test_cases = vec![
            // Case 1: Simple 2x2 matrices
            (
                AirTensor::new(&binding_a_1, vec![2, 2]),
                AirTensor::new(&binding_b_1, vec![2, 2]),
                vec![2, 2],
                vec![3u32; 4], // Expected result: 1 + 2 = 3 for all elements
            ),
            // Case 2: Broadcasting scalar to matrix
            (
                AirTensor::new(&binding_a_2, vec![1]),
                AirTensor::new(&binding_b_2, vec![2, 3]),
                vec![2, 3],
                vec![6u32; 6], // Expected result: 5 + 1 = 6 for all elements
            ),
            // Case 3: Broadcasting row to matrix
            (
                AirTensor::new(&binding_a_3, vec![1, 3]),
                AirTensor::new(&binding_b_3, vec![2, 3]),
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
}
