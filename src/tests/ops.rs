use luminal::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use crate::{binary_test, CairoCompiler};

luminal::test_imports!();

// =============== BINARY ===============

binary_test!(|a, b| a + b, |a, b| a + b, test_add, f32);
