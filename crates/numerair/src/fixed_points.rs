use stwo_prover::core::{
    backend::simd::m31::{PackedBaseField, N_LANES},
    fields::m31::M31,
};

use crate::felt_utils::{felt_to_integer_rep, IntegerRep, P_HALF};

/// The scale (number of fractional bits) in fixed point representation
pub type Scale = u32;

// Constants for fixed-point arithmetic
pub const DEFAULT_SCALE: Scale = 12;
pub const MAX_SCALE: Scale = 20; // Leaves ~10 bits for integer part
pub const MIN_SCALE: Scale = 0;

/// Converts scale to a fixed point multiplier
#[inline]
pub fn scale_to_multiplier(scale: Scale) -> f32 {
    2f32.powi(scale as i32)
}

/// Quantizes a f32 to a field element using fixed point representation
pub fn quantize_float(elem: &f32, shift: f32, scale: Scale) -> Result<IntegerRep, &'static str> {
    let mult = scale_to_multiplier(scale);

    let shifted_elem = *elem + shift;

    // Calculate max value that can be represented
    let max_representable = P_HALF as f32 / mult;

    if shifted_elem.abs() > max_representable {
        return Err("Value overflow in fixed point conversion");
    }

    // Scale the shifted value
    let scaled = (mult * shifted_elem).round() as IntegerRep;

    // Clamp to valid range
    Ok(scaled
        .min(P_HALF as IntegerRep)
        .max(-(P_HALF as IntegerRep)))
}

/// Dequantizes a field element back to f64
pub fn dequantize_to_float(felt: M31, scale: Scale, shift: f32) -> f32 {
    let int_rep = felt_to_integer_rep(felt);
    let multiplier = scale_to_multiplier(scale);
    (int_rep as f32) / multiplier - shift
}

/// Converts a slice of f32 values to packed field elements
pub fn pack_floats(values: &[f32], scale: Scale, shift: f32) -> Vec<PackedBaseField> {
    let n_packed = (values.len() + N_LANES - 1) / N_LANES;

    (0..n_packed)
        .map(|i| {
            let mut lane_values = [0u32; N_LANES];

            for j in 0..N_LANES {
                let idx = i * N_LANES + j;
                if idx < values.len() {
                    lane_values[j] = match quantize_float(&values[idx], shift, scale) {
                        Ok(val) => val as u32,
                        Err(_) => 0, // Handle overflow by setting to 0
                    };
                }
            }

            PackedBaseField::from_array(lane_values.map(M31::from_u32_unchecked))
        })
        .collect()
}

/// Unpacks field elements back to f32 values
pub fn unpack_floats(packed: &[PackedBaseField], scale: Scale, shift: f32, len: usize) -> Vec<f32> {
    packed
        .iter()
        .flat_map(|p| p.to_array())
        .take(len)
        .map(|x| dequantize_to_float(x, scale, shift))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::felt_utils::integer_rep_to_felt;

    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let scale = 12;
        let shift = 0.0;
        let value = 3.14159;

        let quantized = quantize_float(&value, shift, scale).unwrap();
        let felt = integer_rep_to_felt(quantized);
        let dequantized = dequantize_to_float(felt, scale, shift);

        assert!((value - dequantized).abs() < 0.001);
    }

    #[test]
    fn test_pack_unpack() {
        let scale = 12;
        let shift = 0.0;
        let values = vec![1.5f32, 2.5f32, 3.5f32, 4.5f32, 5.5f32];

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert!((original - recovered).abs() < 0.001);
        }
    }

    #[test]
    fn test_pack_unpack_with_shift() {
        let scale = 12;
        let shift = 1.5;
        let values = vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32];
        let precision = 1.0 / scale_to_multiplier(scale);

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        for (i, (original, recovered)) in values.iter().zip(unpacked.iter()).enumerate() {
            println!(
                "Testing value: {}, recovered: {}, error: {}, precision: {}",
                original,
                recovered,
                (original - recovered).abs(),
                precision
            );
            assert!(
                (original - recovered).abs() < precision * 2.0, // Allow slightly more error due to shift
                "Failed at index {}: original={}, recovered={}, error={}, allowed={}",
                i,
                original,
                recovered,
                (original - recovered).abs(),
                precision * 2.0
            );
        }
    }

    #[test]
    fn test_pack_unpack_zero_padding() {
        let scale = 12;
        let shift = 0.0;
        let values = vec![1.0f32, 2.0f32]; // Less than N_LANES elements

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        assert_eq!(values.len(), unpacked.len());
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert!((original - recovered).abs() < 0.001);
        }
    }

    #[test]
    fn test_negative_values() {
        let scale = 12;
        let shift = 0.0;

        // Test batch conversion with mixed positive and negative values
        let mixed_values = vec![-1.5f32, 2.5f32, -3.5f32, 4.5f32, -5.5f32];
        let packed = pack_floats(&mixed_values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, mixed_values.len());

        for (i, (original, recovered)) in mixed_values.iter().zip(unpacked.iter()).enumerate() {
            assert!(
                (original - recovered).abs() < 0.001,
                "Failed at index {}: original={}, recovered={}",
                i,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_overflow() {
        let scale = 20; // High scale to force overflow
        let shift = 0.0;
        let large_value = 1000000.0f32;

        assert!(quantize_float(&large_value, shift, scale).is_err());
    }

    #[test]
    fn test_edge_cases() {
        let scale = 12;
        let shift = 0.0;
        let precision = 1.0 / scale_to_multiplier(scale);

        let tiny_negative = -0.0000001f32;
        let quantized = quantize_float(&tiny_negative, shift, scale).unwrap();
        let felt = integer_rep_to_felt(quantized);
        let dequantized = dequantize_to_float(felt, scale, shift);
        assert_eq!(dequantized, 0.0);

        // Test values close to zero from both sides
        let near_zero_cases = vec![-0.001f32, -0.0001f32, 0.0f32, 0.0001f32, 0.001f32];
        for value in near_zero_cases {
            let quantized = quantize_float(&value, shift, scale).unwrap();
            let felt = integer_rep_to_felt(quantized);
            let dequantized = dequantize_to_float(felt, scale, shift);

            println!(
                "Testing value: {}, quantized: {}, dequantized: {}, precision: {}",
                value, quantized, dequantized, precision
            );

            if value.abs() >= precision {
                assert!(
                    (value - dequantized).abs() <= precision,
                    "Failed for near-zero value {}: got {} after round trip, error larger than one quantization step ({})",
                    value,
                    dequantized,
                    precision
                );
            } else {
                assert_eq!(
                    dequantized, 0.0,
                    "Expected tiny value {} to be quantized to zero, got {}",
                    value, dequantized
                );
            }
        }

        // Test alternating positive/negative values with varying magnitudes
        let alternating = vec![-100.0f32, 0.01f32, -0.01f32, 100.0f32, -1.0f32, 1.0f32];
        let packed = pack_floats(&alternating, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, alternating.len());

        for (i, (original, recovered)) in alternating.iter().zip(unpacked.iter()).enumerate() {
            println!(
                "Index {}: {} -> {}, abs: {}",
                i,
                original,
                recovered,
                original.abs()
            );

            // Calculate allowed error based on value magnitude
            let allowed_error = if original.abs() <= precision {
                // For values smaller than precision, allow zero
                precision
            } else if original.abs() < 0.1 {
                // For small values, allow larger relative error
                precision * 4.0
            } else if original.abs() <= 1.0 {
                // For values around 1.0, allow more error
                precision * 2.0
            } else {
                // For large values, use relative error
                precision * original.abs()
            };

            let actual_error = (original - recovered).abs();
            assert!(
                actual_error <= allowed_error,
                "Failed for alternating value at index {}: original={}, recovered={}, error={}, allowed={}",
                i,
                original,
                recovered,
                actual_error,
                allowed_error
            );
        }
    }

    #[test]
    fn test_symmetry() {
        let scale = 12;
        let shift = 0.0;
        // Test that positive and negative values of the same magnitude have
        // symmetric behavior
        let magnitudes = vec![0.5f32, 1.0f32, 1.5f32, 2.0f32, 10.0f32, 100.0f32];

        for mag in magnitudes {
            let pos = mag;
            let neg = -mag;

            let pos_quantized = quantize_float(&pos, shift, scale).unwrap();
            let pos_felt = integer_rep_to_felt(pos_quantized);
            let neg_quantized = quantize_float(&neg, shift, scale).unwrap();
            let neg_felt = integer_rep_to_felt(neg_quantized);

            let pos_dequantized = dequantize_to_float(pos_felt, scale, shift);
            let neg_dequantized = dequantize_to_float(neg_felt, scale, shift);

            // Check that positive and negative values maintain their magnitude relationship
            assert!(
                (pos_dequantized + neg_dequantized).abs() < 0.001,
                "Asymmetry detected for magnitude {}: pos={}, neg={}",
                mag,
                pos_dequantized,
                neg_dequantized
            );

            // Check precision is similar for both positive and negative
            let pos_error = (pos - pos_dequantized).abs();
            let neg_error = (neg - neg_dequantized).abs();
            assert!(
                (pos_error - neg_error).abs() < 0.001,
                "Precision imbalance detected for magnitude {}",
                mag
            );
        }
    }

    #[test]
    fn test_edge_cases_with_shift() {
        let scale = 12;
        let shift = 1.5;
        let precision = 1.0 / scale_to_multiplier(scale);

        // Test values around the shift point
        let test_values = vec![
            shift - 0.1, // 1.4
            shift,       // 1.5
            shift + 0.1, // 1.6
        ];

        for value in test_values {
            let quantized = quantize_float(&value, shift, scale).unwrap();
            let felt = integer_rep_to_felt(quantized);
            let dequantized = dequantize_to_float(felt, scale, shift);

            println!(
                "Testing value: {}, shifted: {}, quantized: {}, dequantized: {}, precision: {}",
                value,
                value + shift,
                quantized,
                dequantized,
                precision
            );

            assert!(
                (value - dequantized).abs() <= precision * 2.0,  // Allow slightly more error due to shift
                "Failed for value near shift point {}: got {} after round trip, error: {}, allowed: {}",
                value,
                dequantized,
                (value - dequantized).abs(),
                precision * 2.0
            );
        }
    }
}
