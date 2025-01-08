use crate::felt_utils::{felt_to_integer_rep, integer_rep_to_felt, IntegerRep, P_HALF};
use stwo_prover::core::{backend::simd::m31::PackedBaseField, fields::m31::M31};

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

// TODO: Optimize float packing strategy.
// Currently, the same value is broadcast to the entire lane. 
// To benefit of SIMD, we should populate the lane with distinct values, 
// which allows for more efficient utilization of the packing process.
// Note: This requires careful handling of broadcasting rules during tensor operations 
// and ensuring the correct identity value is used based on the specific operation being performed.
pub fn pack_floats(values: &[f32], scale: Scale, shift: f32) -> Vec<PackedBaseField> {
    let mut packed = Vec::new();
    for value in values {
        packed.push(PackedBaseField::broadcast(integer_rep_to_felt(
            quantize_float(value, shift, scale).expect("Quantization failed"),
        )));
    }
    packed
}

pub fn unpack_floats(packed: &[PackedBaseField], scale: Scale, shift: f32, len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    for &packed_value in packed.iter() {
        result.push(dequantize_to_float(
            packed_value.to_array()[0],
            scale,
            shift,
        ));
        if result.len() == len {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
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
        let values = vec![1.5f32, 2.5f32, 3.5f32];

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        assert_eq!(values.len(), unpacked.len());
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert!((original - recovered).abs() < 0.001);
        }
    }

    #[test]
    fn test_pack_unpack_with_shift() {
        let scale = 12;
        let shift = 1.5;
        let values = vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32];

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        assert_eq!(values.len(), unpacked.len());
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            let precision = 1.0 / scale_to_multiplier(scale);
            assert!((original - recovered).abs() < precision * 2.0);
        }
    }

    #[test]
    fn test_negative_values() {
        let scale = 12;
        let shift = 0.0;
        let values = vec![-1.5f32, 2.5f32, -3.5f32, 4.5f32];

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        assert_eq!(values.len(), unpacked.len());
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert!((original - recovered).abs() < 0.001);
        }
    }

    #[test]
    fn test_overflow() {
        let scale = 20;
        let shift = 0.0;
        let large_value = 1000000.0f32;

        assert!(quantize_float(&large_value, shift, scale).is_err());
    }

    #[test]
    fn test_small_values() {
        let scale = 12;
        let shift = 0.0;
        let values = vec![0.0001f32, -0.0001f32, 0.0f32];

        let packed = pack_floats(&values, scale, shift);
        let unpacked = unpack_floats(&packed, scale, shift, values.len());

        let precision = 1.0 / scale_to_multiplier(scale);
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            if original.abs() < precision {
                assert_eq!(*recovered, 0.0);
            } else {
                assert!((original - recovered).abs() < precision);
            }
        }
    }
}
