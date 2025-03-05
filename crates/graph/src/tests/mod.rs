use luminal::{graph::Graph, op::Operator};
use rand::Rng;

mod ops;

#[macro_export]
macro_rules! single_binary_test {
    ($func:expr, $name:ident, $type:ty, ($a_rows:expr, $a_cols:expr), ($b_rows:expr, $b_cols:expr)) => {
        paste::paste! {
            #[test]
            fn [<$name _ $a_rows x $a_cols _ $b_rows x $b_cols>]() {
                let mut rng = StdRng::seed_from_u64(42);
                let a_data = random_vec_rng($a_rows * $a_cols, &mut rng);
                let b_data = random_vec_rng($b_rows * $b_cols, &mut rng);

                // Graph setup
                let mut cx = Graph::new();
                let a = cx.tensor(($a_rows, $a_cols)).set(a_data.clone());
                let b = cx.tensor(($b_rows, $b_cols)).set(b_data.clone());
                let f: fn(GraphTensor, GraphTensor) -> GraphTensor = $func;
                let mut c = f(a, b).retrieve();

                // Compilation and execution using StwoCompiler
                let _ = cx.compile(<(GenericCompiler, StwoCompiler)>::default(), &mut c);
                let trace = cx.gen_trace();
                let proof = cx.prove(trace).expect("Proof generation failed");
                cx.verify(proof).expect("Proof verification failed");
                // Retrieve output data
                let stwo_output = c.data();

                // CPUCompiler comparison
                let mut cx_cpu = Graph::new();
                let a_cpu = cx_cpu.tensor(($a_rows, $a_cols)).set(a_data);
                let b_cpu = cx_cpu.tensor(($b_rows, $b_cols)).set(b_data);
                let mut c_cpu = f(a_cpu, b_cpu).retrieve();
                let _ = cx_cpu.compile(<(GenericCompiler, CPUCompiler)>::default(), &mut c_cpu);
                cx_cpu.execute();
                // Retrieve CPU output
                let cpu_output = c_cpu.data();

                // Assert outputs are close
                assert_close(&stwo_output, &cpu_output);
            }
        }
    };
}

#[macro_export]
macro_rules! binary_test {
    ($func: expr, $name: ident, $type: ty) => {
        // Test operation with same-sized tensors
        $crate::single_binary_test!($func, $name, $type, (3, 4), (3, 4));
        // Test with large tensors to ensure scalability
        $crate::single_binary_test!($func, $name, $type, (32, 32), (32, 32));
        // Test with tensors that have uneven dimensions
        $crate::single_binary_test!($func, $name, $type, (17, 13), (17, 13));

        // TODO(@raphaelDkn): fix broadcasting rules.
        // Test broadcasting a scalar (1,1) to a larger tensor
        // $crate::single_binary_test!($func, $name, $type, (1, 1), (5, 5));
        // Test broadcasting a row vector to a matrix
        // $crate::single_binary_test!($func, $name, $type, (1, 4), (3, 4));
        // Test broadcasting a column vector to a matrix
        // $crate::single_binary_test!($func, $name, $type, (3, 1), (3, 4));
    };
}

#[allow(dead_code)]
pub fn assert_op_in_graph<T: Operator + 'static>(graph: &Graph) {
    assert!(
        graph.node_indices().any(|i| graph.check_node_type::<T>(i)),
        "Node not found in the graph!"
    );
}

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
}

/// Ensure two arrays are nearly equal
pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

/// Ensure two arrays are nearly equal to a decimal place
pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}
