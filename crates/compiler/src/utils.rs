use numerair::backend::cpu::FixedM31;
use stwo_prover::core::backend::simd::m31::PackedBaseField;

// TODO: Optimize float packing strategy.
// Currently, the same value is broadcast to the entire lane.
// To benefit of SIMD, we should populate the lane with distinct values,
// which allows for more efficient utilization of the packing process.
// Note: This requires careful handling of broadcasting rules during tensor operations
// and ensuring the correct identity value is used based on the specific operation being performed.
pub fn pack_floats(values: &[f32]) -> Vec<PackedBaseField> {
    let mut packed = Vec::new();
    for value in values {
        packed.push(PackedBaseField::broadcast(FixedM31::new(*value as f64).0));
    }
    packed
}

pub fn unpack_floats(packed: &[PackedBaseField], len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);
    for &packed_value in packed.iter() {
        result.push(FixedM31::to_f64(&FixedM31(packed_value.to_array()[0])) as f32);
        if result.len() == len {
            break;
        }
    }
    result
}
