use luminal::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use crate::{binary_test, CairoCompiler};

luminal::test_imports!();

// =============== BINARY ===============

binary_test!(|a, b| a + b, |a, b| a + b, test_add, f32);
binary_test!(|a, b| a - b, |a, b| a - b, test_sub, f32);

binary_test!(|a, b| a * b, |a, b| a * b, test_mul, f32);
binary_test!(
    |a, b| a % b,
    |a, b| a.clone() - ((a / b.clone()).to_dtype::<i32>().to_dtype::<f32>() * b),
    test_mod,
    f32
);
binary_test!(|a, b| a.min(b), |a, b| a.minimum(b), test_min, f32);

#[test]
fn test_sum_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(4 * 4096);
    let a = cx.tensor((1, 4, 4096));
    a.set(data.clone());
    let mut b = a.sum_reduce(1).retrieve();
    let mut c = a.sum_reduce(0).retrieve();
    let mut d = a.sum_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<4>, DConst::<4096>));
    let d_b = d_a.clone().sum::<_, DAxis<1>>();
    let d_c = d_a.clone().sum::<_, DAxis<0>>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}