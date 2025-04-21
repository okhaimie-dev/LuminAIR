use super::{assert_close, random_vec_rng};
use crate::graph::LuminairGraph;
use crate::StwoCompiler;
use crate::{binary_test, unary_test};
use luminal::prelude::*;
use luminal_cpu::CPUCompiler;
use rand::{rngs::StdRng, SeedableRng};

// The tests are inspired by Luminal's CUDA tests:
// https://github.com/raphaelDkhn/luminal/blob/main/crates/luminal_cuda/src/tests/fp32.rs

// =============== UNARY ===============
unary_test!(|a| a.recip(), test_recip, f32, true);
unary_test!(|a| a.log2(), test_log2, f32, true);

// =============== BINARY ===============

binary_test!(|a, b| a + b, test_add, f32, false);
binary_test!(|a, b| a * b, test_mul, f32, false);

// =============== REDUCE ===============

#[test]
fn test_sum_reduce() {
    // Graph setup
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(1);
    let data = random_vec_rng(4 * 100, &mut rng, false);
    let a = cx.tensor((1, 4, 100));
    a.set(data.clone());
    let mut b = a.sum_reduce(1).retrieve();
    let mut c = a.sum_reduce(0).retrieve();
    let mut d = a.sum_reduce(2).retrieve();

    // Compilation and execution using StwoCompiler
    cx.compile(
        <(GenericCompiler, StwoCompiler)>::default(),
        (&mut b, &mut c, &mut d),
    );
    let trace = cx.gen_trace().expect("Trace generation failed");
    let proof = cx.prove(trace).expect("Proof generation failed");
    cx.verify(proof).expect("Proof verification failed");

    // CPUCompiler comparison
    let mut cx_cpu = Graph::new();
    let a_cpu = cx_cpu.tensor((1, 4, 100)).set(data.clone());
    let b_cpu = a_cpu.sum_reduce(1).retrieve();
    let c_cpu = a_cpu.sum_reduce(0).retrieve();
    let d_cpu = a_cpu.sum_reduce(2).retrieve();
    cx_cpu.compile(
        <(GenericCompiler, CPUCompiler)>::default(),
        (&mut b, &mut c, &mut d),
    );
    cx_cpu.execute();

    // Assert outputs are close
    assert_close(&b.data(), &b_cpu.data());
    assert_close(&c.data(), &c_cpu.data());
    assert_close(&d.data(), &d_cpu.data());
}

#[test]
fn test_max_reduce() {
    // Graph setup
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(1);
    let data = random_vec_rng(4 * 100, &mut rng, false);
    let a = cx.tensor((1, 4, 100));
    a.set(data.clone());
    let mut b = a.max_reduce(1).retrieve();
    let mut c = a.max_reduce(0).retrieve();
    let mut d = a.max_reduce(2).retrieve();

    // Compilation and execution using StwoCompiler
    cx.compile(
        <(GenericCompiler, StwoCompiler)>::default(),
        (&mut b, &mut c, &mut d),
    );
    let trace = cx.gen_trace().expect("Trace generation failed");
    let proof = cx.prove(trace).expect("Proof generation failed");
    cx.verify(proof).expect("Proof verification failed");

    // CPUCompiler comparison
    let mut cx_cpu = Graph::new();
    let a_cpu = cx_cpu.tensor((1, 4, 100)).set(data.clone());
    let b_cpu = a_cpu.max_reduce(1).retrieve();
    let c_cpu = a_cpu.max_reduce(0).retrieve();
    let d_cpu = a_cpu.max_reduce(2).retrieve();
    cx_cpu.compile(
        <(GenericCompiler, CPUCompiler)>::default(),
        (&mut b, &mut c, &mut d),
    );
    cx_cpu.execute();

    // Assert outputs are close
    assert_close(&b.data(), &b_cpu.data());
    assert_close(&c.data(), &c_cpu.data());
    assert_close(&d.data(), &d_cpu.data());
}
