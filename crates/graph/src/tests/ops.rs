use super::{assert_close, random_vec_rng};
use crate::graph::LuminairGraph;
use crate::StwoCompiler;
use crate::{binary_test, unary_test};
use luminal::prelude::*;
use luminal_cpu::CPUCompiler;
use rand::{rngs::StdRng, SeedableRng};

// =============== UNARY ===============
unary_test!(|a| a.recip(), test_recip, f32);

// =============== BINARY ===============
binary_test!(|a, b| a + b, test_add, f32);
binary_test!(|a, b| a * b, test_mul, f32);
