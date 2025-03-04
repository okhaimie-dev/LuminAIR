use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
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
fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add Operator");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    // Define the sizes to benchmark
    let size = (32, 32);
    let size_str = "(32, 32)";

    // Trace generation benchmark
    group.bench_with_input(
        BenchmarkId::new("Trace Generation", size_str),
        &size,
        |b, &_size| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a + b, (32, 32), (32, 32));
                let _trace = graph.gen_trace();
            })
        },
    );

    // Proof generation benchmark
    group.bench_with_input(BenchmarkId::new("Proving", size_str), &size, |b, &_size| {
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
    });

    // Verification benchmark
    group.bench_with_input(
        BenchmarkId::new("Verification", size_str),
        &size,
        |b, &_size| {
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

    group.finish();
}

// =============== MUL OPERATOR BENCHMARKS ===============
fn benchmark_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Operator");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    // Define the sizes to benchmark
    let size = (32, 32);
    let size_str = "(32, 32)";

    // Trace generation benchmark
    group.bench_with_input(
        BenchmarkId::new("Trace Generation", size_str),
        &size,
        |b, &_size| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a * b, (32, 32), (32, 32));
                let _trace = graph.gen_trace();
            })
        },
    );

    // Proof generation benchmark
    group.bench_with_input(BenchmarkId::new("Proving", size_str), &size, |b, &_size| {
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
    });

    // Verification benchmark
    group.bench_with_input(
        BenchmarkId::new("Verification", size_str),
        &size,
        |b, &_size| {
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

    group.finish();
}

criterion_group!(benches, benchmark_add, benchmark_mul,);
criterion_main!(benches);
