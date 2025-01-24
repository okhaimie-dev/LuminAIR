use luminal::{graph::Graph, op::Operator};
use rand::Rng;

mod ops;

#[macro_export]
macro_rules! single_binary_test {
    ($func: expr, $name: ident, $type: ty, $size: expr) => {
        paste::paste! {
            #[test]
            fn [<$name _ $size>]() {
                let mut rng = StdRng::seed_from_u64(2);
                let a_data = random_vec_rng($size, &mut rng);
                let b_data = random_vec_rng($size, &mut rng);

                // Graph setup
                let mut cx = Graph::new();
                let a = cx.tensor($size).set(a_data.clone());
                let b = cx.tensor($size).set(b_data.clone());
                let f: fn(GraphTensor, GraphTensor) -> GraphTensor = $func;
                let mut c = f(a, b).retrieve();

                // Compilation and execution using StwoCompiler
                let _ = cx.compile(<(GenericCompiler, StwoCompiler)>::default(), &mut c);
                let trace = cx.gen_trace();
                let proof = cx.prove(trace).expect("Proof generation failed");
                cx.verify(proof).expect("Proof verification failed");


                // Retrieve data from `c`
                let stwo_output = cx.get_final_output(c.id);

                // CPUCompiler comparison
                let mut cx_cpu = Graph::new();
                let a_cpu = cx_cpu.tensor($size).set(a_data.clone());
                let b_cpu = cx_cpu.tensor($size).set(b_data.clone());
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
    ($func: expr , $name: ident, $type: ty) => {
        $crate::single_binary_test!($func, $name, $type, 3);
        $crate::single_binary_test!($func, $name, $type, 50);
        $crate::single_binary_test!($func, $name, $type, 783);
        $crate::single_binary_test!($func, $name, $type, 4096);
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
