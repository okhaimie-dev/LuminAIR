use criterion::{criterion_group, criterion_main, Criterion, PlotConfiguration};
use luminair_graph::{graph::LuminairGraph, StwoCompiler};
use luminal::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

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

// Define a benchmark parameter that combines operation and tensor size
#[derive(Debug, Clone, Copy)]
enum Stage {
    TraceGeneration,
    Proving,
    Verification,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Stage::TraceGeneration => write!(f, "Trace Generation"),
            Stage::Proving => write!(f, "Proving"),
            Stage::Verification => write!(f, "Verification"),
        }
    }
}

struct BenchParams {
    stage: Stage,
    size: (usize, usize),
}

impl fmt::Display for BenchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({}x{})", self.stage, self.size.0, self.size.1)
    }
}

// Benchmark for Add operator
fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add Operator");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    let sizes = [(32, 32)];

    for &size in &sizes {
        let (rows, cols) = size;

        // Trace generation
        let params = BenchParams {
            stage: Stage::TraceGeneration,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a + b, (rows, cols), (rows, cols));
                let _trace = graph.gen_trace();
            })
        });

        // Proof generation
        let params = BenchParams {
            stage: Stage::Proving,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let mut graph = create_graph!(|a, b| a + b, (rows, cols), (rows, cols));
                    let trace = graph.gen_trace();
                    (graph, trace)
                },
                |(mut graph, trace)| {
                    let _proof = graph.prove(trace).expect("Proof generation failed");
                },
            )
        });

        // Verification
        let params = BenchParams {
            stage: Stage::Verification,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let mut graph = create_graph!(|a, b| a + b, (rows, cols), (rows, cols));
                    let trace = graph.gen_trace();
                    let proof = graph.prove(trace).expect("Proof generation failed");
                    (graph, proof)
                },
                |(graph, proof)| {
                    graph.verify(proof).expect("Proof verification failed");
                },
            )
        });
    }

    group.finish();
}

// Benchmark for Mul operator
fn benchmark_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul Operator");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    let sizes = [(32, 32)];

    for &size in &sizes {
        let (rows, cols) = size;

        // Trace generation
        let params = BenchParams {
            stage: Stage::TraceGeneration,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter(|| {
                let mut graph = create_graph!(|a, b| a * b, (rows, cols), (rows, cols));
                let _trace = graph.gen_trace();
            })
        });

        // Proof generation
        let params = BenchParams {
            stage: Stage::Proving,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let mut graph = create_graph!(|a, b| a * b, (rows, cols), (rows, cols));
                    let trace = graph.gen_trace();
                    (graph, trace)
                },
                |(mut graph, trace)| {
                    let _proof = graph.prove(trace).expect("Proof generation failed");
                },
            )
        });

        // Verification
        let params = BenchParams {
            stage: Stage::Verification,
            size,
        };
        group.bench_function(params.to_string(), |b| {
            b.iter_with_setup(
                || {
                    let mut graph = create_graph!(|a, b| a * b, (rows, cols), (rows, cols));
                    let trace = graph.gen_trace();
                    let proof = graph.prove(trace).expect("Proof generation failed");
                    (graph, proof)
                },
                |(graph, proof)| {
                    graph.verify(proof).expect("Proof verification failed");
                },
            )
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_add, benchmark_mul,);
criterion_main!(benches);
