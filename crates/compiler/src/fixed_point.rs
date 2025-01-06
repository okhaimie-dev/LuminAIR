use stwo_prover::core::{
    backend::simd::m31::{PackedBaseField, N_LANES},
    fields::m31::BaseField,
};

// The modulus P for the M31 field
const P: u32 = (1 << 31) - 1;
const P_HALF: u32 = P >> 1;

// Constants for fixed-point arithmetic
pub const DEFAULT_SCALE: i32 = 12;
pub const MAX_SCALE: i32 = 20;
pub const MIN_SCALE: i32 = 0;

/// Converts scale (log base 2) to a fixed point multiplier
pub fn scale_to_multiplier(scale: i32) -> f32 {
    f32::powf(2.0, scale as f32)
}

/// Quantizes a f32 to a field element using fixed point representation
pub fn quantize_float(value: f32, scale: i32) -> Result<u32, &'static str> {
    let multiplier = scale_to_multiplier(scale);

    // Special handling for very small values
    if value.abs() < 1.0 / multiplier {
        return Ok(0);
    }

    let scaled = (multiplier * value).round() as i64;

    if scaled >= (P as i64) || scaled <= -(P as i64) {
        return Err("Value overflow in fixed point conversion");
    }

    let result = if scaled < 0 {
        (P as i64 + scaled) as u32
    } else {
        scaled as u32
    };

    Ok(result % P)
}

/// Dequantizes a field element back to f32
pub fn dequantize_float(value: u32, scale: i32) -> f32 {
    let multiplier = scale_to_multiplier(scale);

    // Interpret values > P/2 as negative
    let signed_val = if value > P_HALF {
        -((P - value) as i64)
    } else {
        value as i64
    };

    (signed_val as f32) / multiplier
}

/// Converts a slice of f32 values to packed field elements
pub fn pack_floats(values: &[f32], scale: i32) -> Vec<PackedBaseField> {
    let n_packed = (values.len() + N_LANES - 1) / N_LANES;

    (0..n_packed)
        .map(|i| {
            let mut lane_values = [0u32; N_LANES];

            for j in 0..N_LANES {
                let idx = i * N_LANES + j;
                lane_values[j] = if idx < values.len() {
                    quantize_float(values[idx], scale).unwrap_or(0)
                } else {
                    0
                };
            }

            PackedBaseField::from_array(lane_values.map(BaseField::from_u32_unchecked))
        })
        .collect()
}

/// Unpacks field elements back to f32 values
pub fn unpack_floats(packed: &[PackedBaseField], scale: i32, len: usize) -> Vec<f32> {
    packed
        .iter()
        .flat_map(|p| p.to_array())
        .take(len)
        .map(|x| dequantize_float(x.0, scale))
        .collect()
}

/// Validates that a scale factor is within acceptable bounds
pub fn validate_scale(scale: i32) -> Result<(), &'static str> {
    if scale < MIN_SCALE || scale > MAX_SCALE {
        Err("Scale factor out of valid range")
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let scale = 12;
        let value = 3.14159f32;

        // Test quantize and dequantize
        let quantized = quantize_float(value, scale).unwrap();
        let dequantized = dequantize_float(quantized, scale);

        assert!((value - dequantized).abs() < 0.001);
    }

    #[test]
    fn test_pack_unpack() {
        let scale = 12;
        let values = vec![1.5f32, 2.5f32, 3.5f32, 4.5f32, 5.5f32];

        let packed = pack_floats(&values, scale);
        let unpacked = unpack_floats(&packed, scale, values.len());

        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert!((original - recovered).abs() < 0.001);
        }
    }

    #[test]
    fn test_overflow() {
        let scale = 20; // High scale to force overflow
        let large_value = 1000000.0f32;

        assert!(quantize_float(large_value, scale).is_err());
    }

    #[test]
    fn test_negative_values() {
        let scale = 12;
        let test_cases = vec![-1.5f32, -2.5f32, -0.000123f32, -123.456f32, -3.14159f32];

        for value in test_cases {
            // Test round trip for single value
            let quantized = quantize_float(value, scale).unwrap();
            let dequantized = dequantize_float(quantized, scale);

            println!("Original: {}, Recovered: {}", value, dequantized);
            assert!(
                (value - dequantized).abs() < 0.001,
                "Failed for value {}: got {} after round trip",
                value,
                dequantized
            );
        }

        // Test batch conversion with mixed positive and negative values
        let mixed_values = vec![-1.5f32, 2.5f32, -3.5f32, 4.5f32, -5.5f32];
        let packed = pack_floats(&mixed_values, scale);
        let unpacked = unpack_floats(&packed, scale, mixed_values.len());

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
    fn test_edge_cases() {
        let scale = 12;
        let precision = 1.0 / scale_to_multiplier(scale);
    
        // Test very small negative values below the precision
        let tiny_negative = -0.0000001f32;
        let quantized = quantize_float(tiny_negative, scale).unwrap();
        let dequantized = dequantize_float(quantized, scale);
        assert_eq!(dequantized, 0.0);
    
        // Test values close to zero from both sides
        let near_zero_cases = vec![-0.001f32, -0.0001f32, 0.0f32, 0.0001f32, 0.001f32];
        for value in near_zero_cases {
            let quantized = quantize_float(value, scale).unwrap();
            let dequantized = dequantize_float(quantized, scale);
            
            println!("Testing value: {}, quantized: {}, dequantized: {}, precision: {}", 
                    value, quantized, dequantized, precision);
    
            if value.abs() >= precision {
                // For values above the precision threshold, check if the error is within one quantization step
                assert!(
                    (value - dequantized).abs() <= precision,
                    "Failed for near-zero value {}: got {} after round trip, error larger than one quantization step ({})",
                    value,
                    dequantized,
                    precision
                );
            } else {
                // For values below the precision threshold, expect them to be quantized to zero
                assert_eq!(
                    dequantized, 
                    0.0,
                    "Expected tiny value {} to be quantized to zero, got {}",
                    value,
                    dequantized
                );
            }
        }
    
        // Test alternating positive/negative values with varying magnitudes
        let alternating = vec![-100.0f32, 0.01f32, -0.01f32, 100.0f32, -1.0f32, 1.0f32];
        let packed = pack_floats(&alternating, scale);
        let unpacked = unpack_floats(&packed, scale, alternating.len());
    
        for (i, (original, recovered)) in alternating.iter().zip(unpacked.iter()).enumerate() {
            println!("Index {}: {} -> {}", i, original, recovered);
            assert!(
                (original - recovered).abs() <= precision.max(precision * original.abs()),
                "Failed for alternating value at index {}: error {} exceeds maximum allowed error {}",
                i,
                (original - recovered).abs(),
                precision.max(precision * original.abs())
            );
        }
    }

    #[test]
    fn test_symmetry() {
        let scale = 12;
        // Test that positive and negative values of the same magnitude
        // symmetric behavior
        let magnitudes = vec![0.5f32, 1.0f32, 1.5f32, 2.0f32, 10.0f32, 100.0f32];

        for mag in magnitudes {
            let pos = mag;
            let neg = -mag;

            let pos_quantized = quantize_float(pos, scale).unwrap();
            let neg_quantized = quantize_float(neg, scale).unwrap();

            let pos_dequantized = dequantize_float(pos_quantized, scale);
            let neg_dequantized = dequantize_float(neg_quantized, scale);

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
}
