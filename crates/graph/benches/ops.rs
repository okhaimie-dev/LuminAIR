use criterion::{criterion_group, criterion_main, Criterion};
use luminair_graph::{graph::LuminairGraph, StwoCompiler};
use luminal::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
}

macro_rules! create_graph {
    ($func:expr, ($a_rows:expr, $a_cols:expr), ($b_rows:expr, $b_cols:expr)) => {{
        let mut rng = StdRng::seed_from_u64(42);
        let a_data = random_vec_rng($a_rows * $a_cols, &mut rng);
        let b_data = random_vec_rng($b_rows * $b_cols, &mut rng);

        // Graph setup
        let mut cx = Graph::new();
        let a = cx.tensor(($a_rows, $a_cols)).set(a_data.clone());
        let b = cx.tensor(($b_rows, $b_cols)).set(b_data.clone());
        let f = $func;
        let mut c = f(a, b).retrieve();

        // Compilation and execution using StwoCompiler
        cx.compile(<(GenericCompiler, StwoCompiler)>::default(), &mut c);

        cx
    }};
}

// =============== ADD OPERATOR BENCHMARKS ===============

/// Benchmark for Add operator: trace generation.
fn benchmark_add_trace_generation(c: &mut Criterion) {
    c.bench_function(
        "Trace Generation of Add operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a + b, (32, 32), (32, 32));
                let _trace = graph.gen_trace();
            })
        },
    );
}

/// Benchmark for Add operator: proof generation.
fn benchmark_add_prove(c: &mut Criterion) {
    c.bench_function(
        "Proving of Add operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter_with_setup(
                || {
                    // Setup: Create graph and generate trace
                    let mut graph = create_graph!(|a, b| a + b, (32, 32), (32, 32));
                    let trace = graph.gen_trace();
                    (graph, trace)
                },
                |(mut graph, trace)| {
                    let _proof = graph.prove(trace).expect("Proof generation failed");
                },
            )
        },
    );
}

/// Benchmark for Add operator: proof verification.
fn benchmark_add_verify(c: &mut Criterion) {
    c.bench_function(
        "Proof verification of Add operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter_with_setup(
                || {
                    // Setup: Create graph, generate trace, and create proof
                    let mut graph = create_graph!(|a, b| a + b, (32, 32), (32, 32));
                    let trace = graph.gen_trace();
                    let proof = graph.prove(trace).expect("Proof generation failed");
                    (graph, proof)
                },
                |(graph, proof)| {
                    graph.verify(proof).expect("Proof verification failed");
                },
            )
        },
    );
}

// =============== MUL OPERATOR BENCHMARKS ===============

/// Benchmark for Mul operator: trace generation.
fn benchmark_mul_trace_generation(c: &mut Criterion) {
    c.bench_function(
        "Trace Generation of Mul operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a * b, (32, 32), (32, 32));
                let _trace = graph.gen_trace();
            })
        },
    );
}

/// Benchmark for Mul operator: proof generation.
fn benchmark_mul_prove(c: &mut Criterion) {
    c.bench_function(
        "Proving of Mul operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter_with_setup(
                || {
                    // Setup: Create graph and generate trace
                    let mut graph = create_graph!(|a, b| a * b, (32, 32), (32, 32));
                    let trace = graph.gen_trace();
                    (graph, trace)
                },
                |(mut graph, trace)| {
                    let _proof = graph.prove(trace).expect("Proof generation failed");
                },
            )
        },
    );
}

/// Benchmark for Mul operator: proof verification.
fn benchmark_mul_verify(c: &mut Criterion) {
    c.bench_function(
        "Proof verification of Mul operator on tensors (32, 32), (32, 32)",
        |b| {
            b.iter_with_setup(
                || {
                    // Setup: Create graph, generate trace, and create proof
                    let mut graph = create_graph!(|a, b| a * b, (32, 32), (32, 32));
                    let trace = graph.gen_trace();
                    let proof = graph.prove(trace).expect("Proof generation failed");
                    (graph, proof)
                },
                |(graph, proof)| {
                    graph.verify(proof).expect("Proof verification failed");
                },
            )
        },
    );
}

criterion_group!(
    benches,
    benchmark_add_trace_generation,
    benchmark_add_prove,
    benchmark_add_verify,
    benchmark_mul_trace_generation,
    benchmark_mul_prove,
    benchmark_mul_verify,
);
criterion_main!(benches);
