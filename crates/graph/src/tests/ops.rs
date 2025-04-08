use super::{assert_close, random_vec_rng};
use crate::graph::LuminairGraph;
use crate::StwoCompiler;
use crate::{binary_test, unary_test};
use luminal::prelude::*;
use luminal_cpu::CPUCompiler;
use rand::{rngs::StdRng, SeedableRng};

// =============== UNARY ===============
unary_test!(|a| a.recip(), test_recip, f32, true);
unary_test!(|a| a.sum_reduce(0), test_sum_reduce, f32, true);


fn sum_reduce_wrapper_axes_0(a: GraphTensor, _b: GraphTensor) -> GraphTensor {
    a.sum_reduce(0) 
}

fn sum_reduce_wrapper_axes_1(a: GraphTensor, _b: GraphTensor) -> GraphTensor {
    a.sum_reduce(1) 
}

// Use binary_test with the wrapper function
binary_test!(sum_reduce_wrapper_axes_0, test_sum_reduce_0, f32, false);
binary_test!(sum_reduce_wrapper_axes_1, test_sum_reduce_1, f32, false);

// =============== BINARY ===============

binary_test!(|a, b| a + b, test_add, f32, false);
binary_test!(|a, b| a * b, test_mul, f32, false);
