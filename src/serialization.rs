use cairo1_run::FuncArg;
use cairo_vm::Felt252;
use num_traits::FromPrimitive;

use crate::fixed_point::from_float_to_fp;

pub(crate) fn serialize_inputs_binary_op(lhs: Vec<f32>, rhs: Vec<f32>) -> Vec<FuncArg> {
    let mut serialized: Vec<FuncArg> = Vec::with_capacity(2);

    let serialized_lhs = FuncArg::Array(
        lhs.into_iter()
            .map(|e| Felt252::from_i64(from_float_to_fp(e)).unwrap())
            .collect(),
    );
    let serialized_rhs = FuncArg::Array(
        rhs.into_iter()
            .map(|e| Felt252::from_i64(from_float_to_fp(e)).unwrap())
            .collect(),
    );
    serialized.push(serialized_lhs);
    serialized.push(serialized_rhs);

    serialized
}
