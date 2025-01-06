use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use luminal::prelude::*;

use crate::data::StwoData;
use luminair_air::{
    backend::simd::add::trace::generate_trace, tensor::AirTensor, utils::calculate_log_size,
};

#[derive(Debug, Default)]
pub struct PrimitiveCompiler {}
impl PrimitiveCompiler {
    pub fn new() -> Self {
        Self {}
    }
}

// ====== BINARY ======
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StwoAdd;
impl Operator for StwoAdd {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp.len() != 2 {
            panic!("Add operator requires exactly two input tensors.");
        }

        let (a_tensor, a_shape) = &inp[0];
        let (b_tensor, b_shape) = &inp[1];

        let get_data = |tensor: &InputTensor| {
            if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
                StwoData::from_f32(data)
            } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
                StwoData(Arc::clone(&data.0))
            } else {
                panic!("Unsupported input type for StwoAdd");
            }
        };

        let a_data = get_data(a_tensor);
        let b_data = get_data(b_tensor);

        // Create AirTensors
        let a = AirTensor::new(a_data.as_slice(), a_shape.shape_usize());
        let b = AirTensor::new(b_data.as_slice(), b_shape.shape_usize());

        // Calculate required trace size based on tensor dimensions
        let max_size = a.size().max(b.size());
        let required_log_size = calculate_log_size(max_size);

        // Generate trace and get result tensor
        let (_trace, c) = generate_trace(required_log_size, &a, &b);

        let c = vec![Tensor::new(StwoData(Arc::new(c.data().to_vec())))];
        println!("Output: {:?}", c);
        c
    }
}

impl Compiler for PrimitiveCompiler {
    type Output = ();

    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _ids: T) -> Self::Output {
        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        for id in graph.node_indices().collect::<Vec<_>>() {
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Add>(op) {
                *op_ref = Box::new(StwoAdd)
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else if is::<Function>(op) {
                // Keep the Function operator as is
                continue;
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
