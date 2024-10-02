use luminal::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use crate::{binary_test, unary_test, CairoCompiler};

luminal::test_imports!();

// =============== UNARY ===============

unary_test!(|a| a.sin(), |a| a.sin(), test_sin, f32);
unary_test!(|a| a.sqrt(), |a| a.sqrt(), test_sqrt, f32);
unary_test!(|a| a.recip(), |a| a.recip(), test_recip, f32);
unary_test!(|a| a * a, |a| a.clone() * a, test_square, f32);
unary_test!(|a| a.ln(), |a| a.ln(), test_ln, f32);
unary_test!(|a| a.log2(), |a| a.ln() / 2_f32.ln(), test_log2, f32);
unary_test!(|a| a.exp2(), |a| (a * 2_f32.ln()).exp(), test_exp2, f32);
// unary_test!(
//     |a| a.softmax(0),
//     |a| a.softmax::<DAxis<0>>(),
//     test_softmax,
//     f32
// );
unary_test!(
    |a| a.mean_norm(0).std_norm(0, 1e-5),
    |a| a.normalize::<DAxis<0>>(1e-5),
    test_norm,
    f32
);

// =============== BINARY ===============

binary_test!(|a, b| a + b, |a, b| a + b, test_add, f32);
binary_test!(|a, b| a - b, |a, b| a - b, test_sub, f32);
binary_test!(|a, b| a * b, |a, b| a * b, test_mul, f32);
binary_test!(|a, b| a / b, |a, b| a / b, test_div, f32);
binary_test!(
    |a, b| a % b,
    |a, b| a.clone() - ((a / b.clone()).to_dtype::<i32>().to_dtype::<f32>() * b),
    test_mod,
    f32
);
binary_test!(|a, b| a.min(b), |a, b| a.minimum(b), test_min, f32);
binary_test!(|a, b| a.max(b), |a, b| a.maximum(b), test_max, f32);

// =============== MOVEMENT ===============

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((3, 4)).set(data.clone());
    let mut b = a.permute((1, 0)).reshape((12, 1)).retrieve();
    let _ = cx.compile(CairoCompiler::default(), &mut b);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<3>, DConst::<4>));
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.as_vec());
}

// =============== REDUCE ===============

#[test]
fn test_sum_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(4 * 512);
    let a = cx.tensor((1, 4, 512));
    a.set(data.clone());
    let mut b = a.sum_reduce(1).retrieve();
    let mut c = a.sum_reduce(0).retrieve();
    let mut d = a.sum_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut c, &mut d));
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<4>, DConst::<512>));
    let d_b = d_a.clone().sum::<_, DAxis<1>>();
    let d_c = d_a.clone().sum::<_, DAxis<0>>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_max_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((2, 2, 3));
    a.set(data.clone());
    let mut b = a.max_reduce(1).retrieve();
    let mut c = a.max_reduce(0).retrieve();
    let mut d = a.max_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut c, &mut d));
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<2>, DConst::<2>, DConst::<3>));
    let d_b = d_a.clone().max::<_, DAxis<1>>();
    let d_c = d_a.clone().max::<_, DAxis<0>>();
    let d_d = d_a.max::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

// #[test]
// fn test_mean_reduce() {
//     let data = random_vec(512);
//     let mut cx = Graph::new();
//     let a = cx.tensor((1, 10, 512)).set(data.clone());
//     let mut b = a.mean_reduce(2).retrieve();

//     let _ = cx.compile(CairoCompiler::default(), &mut b);
//     cx.execute();

//     let d_dev = Cpu::default();
//     let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<512>));
//     let d_b = d_a.mean::<_, DAxis<2>>();
//     assert_close(&b.data(), &d_b.as_vec());
// }

// =============== MATMUL ===============

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a_data = random_vec(3 * 3);
    let b_data = random_vec(3 * 3);
    let a = cx.tensor((3, 3)).set(a_data.clone());
    let b = cx.tensor((3, 3)).set(b_data.clone());
    let mut c = a.matmul(b).retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<3>, DConst::<3>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<3>, DConst::<3>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}
