#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_helpers::*;
    use ark_ff::Field;
    use fixed::types::I64F64;
    use fixed_macro::fixed;

    #[test]
    fn test_log2_precision() {
        let input_values = vec![
            fixed!(2.0: I64F64),
            fixed!(4.0: I64F64),
            fixed!(8.0: I64F64),
            fixed!(16.0: I64F64),
            fixed!(1.5: I64F64),
            fixed!(3.75: I64F64),
        ];

        for input_val in input_values {
            let mut eval = Log2Eval::new();
            let input = input_val.to_bits().into();
            let output = eval.compute_output(&input);
            
            // Convert back to fixed point for comparison
            let output_fixed = I64F64::from_bits(output.into_bigint().try_into().unwrap());
            let expected = input_val.log2();
            
            // Check precision (within 2^-32)
            assert!((output_fixed - expected).abs() < fixed!(0.0000000001: I64F64),
                "Log2 precision test failed for input {}: got {}, expected {}",
                input_val, output_fixed, expected);
            
            // Verify AIR constraint: 2^output = input
            let pow2_result = fixed!(2.0: I64F64).pow(output_fixed);
            assert!((pow2_result - input_val).abs() < fixed!(0.0000000001: I64F64),
                "AIR constraint test failed for input {}: 2^{} = {} ≠ {}",
                input_val, output_fixed, pow2_result, input_val);
        }
    }

    #[test]
    fn test_log2_edge_cases() {
        let mut eval = Log2Eval::new();
        
        // Test value very close to 1
        let near_one = fixed!(1.000000001: I64F64);
        let input = near_one.to_bits().into();
        let output = eval.compute_output(&input);
        let output_fixed = I64F64::from_bits(output.into_bigint().try_into().unwrap());
        
        assert!((output_fixed).abs() < fixed!(0.0000000001: I64F64),
            "Log2 of near 1 should be very close to 0");
            
        // Test minimum positive value that can be represented
        let min_positive = I64F64::from_bits(1);
        let input = min_positive.to_bits().into();
        let output = eval.compute_output(&input);
        let output_fixed = I64F64::from_bits(output.into_bigint().try_into().unwrap());
        
        // The result should be negative but finite
        assert!(output_fixed < fixed!(0.0: I64F64),
            "Log2 of minimum positive value should be negative");
    }
} 