use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use luminair_air::{
    ops::{add::TensorAdd, mul::TensorMul},
    tensor::AirTensor,
    Circuit,
};
use numerair::FixedPoint;
use stwo_prover::core::{
    backend::simd::m31::{PackedBaseField, LOG_N_LANES},
    pcs::PcsConfig,
    vcs::blake2_merkle::Blake2sMerkleChannel,
};

fn get_test_cases() -> Vec<(
    String,
    AirTensor<'static, PackedBaseField>,
    AirTensor<'static, PackedBaseField>,
    u32,
)> {
    vec![
        // Small matrices (2x2)
        (
            "2x2_2x2".to_string(),
            AirTensor::from_vec(vec![PackedBaseField::from_f64(1.); 4], vec![2, 2]),
            AirTensor::from_vec(vec![PackedBaseField::from_f64(2.); 4], vec![2, 2]),
            LOG_N_LANES + 2,
        ),
        // Medium matrices (50x50)
        (
            "50x50_50x1".to_string(),
            AirTensor::from_vec(vec![PackedBaseField::from_f64(1.); 50 * 50], vec![50, 50]),
            AirTensor::from_vec(vec![PackedBaseField::from_f64(2.); 50], vec![50, 1]),
            LOG_N_LANES + 2,
        ),
        // Large matrices (100x100)
        (
            "100x100_100x1".to_string(),
            AirTensor::from_vec(
                vec![PackedBaseField::from_f64(1.); 100 * 100],
                vec![100, 100],
            ),
            AirTensor::from_vec(vec![PackedBaseField::from_f64(2.); 100], vec![100, 1]),
            LOG_N_LANES + 10,
        ),
    ]
}

fn bench_tensor_add(c: &mut Criterion) {
    let config = PcsConfig::default();
    let test_cases = get_test_cases();

    // Benchmark trace generation
    {
        let mut group = c.benchmark_group("TensorAdd/tracing");
        group.measurement_time(std::time::Duration::from_secs(10));
        group.sample_size(10);
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorAdd {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            group.bench_with_input(BenchmarkId::from_parameter(name), &circuit, |b, circuit| {
                b.iter(|| circuit.generate_trace());
            });
        }
        group.finish();
    }

    // Benchmark proving
    {
        let mut group = c.benchmark_group("TensorAdd/proving");
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorAdd {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            let (trace, _c) = circuit.generate_trace();

            group.bench_with_input(BenchmarkId::from_parameter(name), &trace, |b, trace| {
                b.iter(|| TensorAdd::prove::<Blake2sMerkleChannel>(trace, config));
            });
        }
        group.finish();
    }

    // Benchmark verification
    {
        let mut group = c.benchmark_group("TensorAdd/verification");
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorAdd {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            let (trace, _c) = circuit.generate_trace();

            group.bench_with_input(BenchmarkId::from_parameter(name), &trace, |b, trace| {
                b.iter_with_setup(
                    || TensorAdd::prove::<Blake2sMerkleChannel>(trace, config),
                    |(components, proof)| {
                        TensorAdd::verify::<Blake2sMerkleChannel>(components, proof, config)
                    },
                )
            });
        }
        group.finish();
    }
}

fn bench_tensor_mul(c: &mut Criterion) {
    let config = PcsConfig::default();
    let test_cases = get_test_cases();

    // Benchmark trace generation
    {
        let mut group = c.benchmark_group("TensorMul/tracing");
        group.measurement_time(std::time::Duration::from_secs(10));
        group.sample_size(10);
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorMul {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            group.bench_with_input(BenchmarkId::from_parameter(name), &circuit, |b, circuit| {
                b.iter(|| circuit.generate_trace());
            });
        }
        group.finish();
    }

    // Benchmark proving
    {
        let mut group = c.benchmark_group("TensorMul/proving");
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorMul {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            let (trace, _c) = circuit.generate_trace();

            group.bench_with_input(BenchmarkId::from_parameter(name), &trace, |b, trace| {
                b.iter(|| TensorMul::prove::<Blake2sMerkleChannel>(trace, config));
            });
        }
        group.finish();
    }

    // Benchmark verification
    {
        let mut group = c.benchmark_group("TensorMul/verification");
        for (name, tensor_a, tensor_b, log_size) in test_cases.iter() {
            let circuit = TensorMul {
                a: tensor_a,
                b: tensor_b,
                log_size: *log_size,
            };
            let (trace, _c) = circuit.generate_trace();

            group.bench_with_input(BenchmarkId::from_parameter(name), &trace, |b, trace| {
                b.iter_with_setup(
                    || TensorMul::prove::<Blake2sMerkleChannel>(trace, config),
                    |(components, proof)| {
                        TensorMul::verify::<Blake2sMerkleChannel>(components, proof, config)
                    },
                )
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_tensor_add, bench_tensor_mul);
criterion_main!(benches);
