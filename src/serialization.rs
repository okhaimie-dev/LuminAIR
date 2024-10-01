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

pub(crate) fn serialize_reduce_op(
    input: &Vec<f32>,
    front_size: usize,
    back_size: usize,
    dim_size: usize,
) -> Vec<FuncArg> {
    let mut serialized: Vec<FuncArg> = Vec::with_capacity(2);

    let serialized_input = FuncArg::Array(
        input
            .into_iter()
            .map(|e| Felt252::from_i64(from_float_to_fp(*e)).unwrap())
            .collect(),
    );

    serialized.push(serialized_input);
    serialized.push(FuncArg::Single(Felt252::from_usize(front_size).unwrap()));
    serialized.push(FuncArg::Single(Felt252::from_usize(back_size).unwrap()));
    serialized.push(FuncArg::Single(Felt252::from_usize(dim_size).unwrap()));

    serialized
}
