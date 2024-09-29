use std::{
    any::{Any, TypeId},
    path::PathBuf,
    sync::Arc,
};

use luminal::prelude::*;

use crate::{
    cairo_runner::CairoRunnerConfig,
    precomputing::{compute_strides, determine_broadcast_shape, expand_data},
    CairoCompilerError,
};
use itertools::Itertools;

#[derive(Clone)]
pub struct CairoAdd {
    sierra_file: PathBuf,
    runner_config: Arc<CairoRunnerConfig>,
}
crate::debug_type!(CairoAdd);

impl CairoAdd {
    pub fn new(sierra_file: PathBuf, runner_config: Arc<CairoRunnerConfig>) -> Self {
        Self {
            sierra_file,
            runner_config,
        }
    }
}

impl Operator for CairoAdd {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Ensure exactly two input tensors
        if tensors.len() != 2 {
            panic!("CairoAdd operator requires exactly two input tensors.");
        }

        // Extract input tensors and their ShapeTrackers
        let (tensor_a, shape_a_tracker) = &tensors[0];
        let (tensor_b, shape_b_tracker) = &tensors[1];

        // Downcast to Vec<f32>
        let data_a = get_vec(tensor_a);
        let data_b = get_vec(tensor_b);

        // Get original shapes
        let shape_a = shape_a_tracker.shape_usize();
        let shape_b = shape_b_tracker.shape_usize();

        // Determine broadcasted shape
        let broadcast_shape = match determine_broadcast_shape(&shape_a, &shape_b) {
            Ok(shape) => shape,
            Err(e) => panic!("Broadcasting error: {}", e),
        };

        // Compute strides for original tensors
        let strides_a = compute_strides(&shape_a);
        let strides_b = compute_strides(&shape_b);

        // Expand data according to broadcasted shape
        let expanded_a = expand_data(&data_a, &shape_a, &broadcast_shape, &strides_a);
        let expanded_b = expand_data(&data_b, &shape_b, &broadcast_shape, &strides_b);

        // Perform element-wise addition
        let result_data: Vec<f32> = expanded_a
            .iter()
            .zip(expanded_b.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Return the result tensor
        vec![Tensor::new(result_data)]
    }
}

/// Convert all primitive ops to cairo primitive ops.
#[derive(Debug, Default)]
pub struct PrimitiveCompiler();

impl Compiler for PrimitiveCompiler {
    type Output = Result<(), CairoCompilerError>;

    fn compile<T: luminal::prelude::ToIdsMut>(
        &self,
        graph: &mut luminal::prelude::Graph,
        ids: T,
    ) -> Self::Output {
        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        // Swap primitive ops
        for id in graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Log2>(op) {
                unimplemented!()
            } else if is::<Exp2>(op) {
                unimplemented!()
            } else if is::<Sin>(op) {
                unimplemented!()
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                unimplemented!()
            } else if is::<Recip>(op) {
                unimplemented!()
            } else if is::<Sqrt>(op) {
                unimplemented!()
            } else if is::<Add>(op) {
                unimplemented!()
            } else if is::<Mul>(op) {
                unimplemented!()
            } else if is::<Mod>(op) {
                unimplemented!()
            } else if is::<LessThan>(op) {
                unimplemented!()
            } else if is::<Contiguous>(op) {
                unimplemented!()
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                unimplemented!()
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                unimplemented!()
            }
        }

        todo!()
    }
}

/// Helper function to extract Vec<f32> from InputTensor
fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor
        .borrowed()
        .downcast_ref::<Vec<f32>>()
        .expect("Tensor data is not Vec<f32>")
}
