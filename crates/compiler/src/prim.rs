use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use luminair_air::{calculate_log_size, gen_add_trace};
use luminal::prelude::*;

use crate::data::StwoData;

#[derive(Default)]
pub struct PrimitiveCompiler {
    config: Arc<Config>,
}
impl PrimitiveCompiler {
    pub fn new(config: Config) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Config {}

// ====== BINARY ======
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add {
    node_id: usize,
    config: Arc<Config>,
}

impl Add {
    pub fn new(node_id: usize, config: Arc<Config>) -> Self {
        Self { node_id, config }
    }
}

impl Operator for Add {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp.len() != 2 {
            panic!("Add operator requires exactly two input tensors.");
        }

        // Get data
        let (a_tensor, _a_shape) = &inp[0];
        let (b_tensor, _b_shape) = &inp[1];

        let get_data = |tensor: &InputTensor| {
            if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
                StwoData::from_f32(data)
            } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
                StwoData(Arc::clone(&data.0))
            } else {
                panic!("Unsupported input type for Add");
            }
        };

        let a = get_data(a_tensor);
        let b = get_data(b_tensor);

        // Calculate required trace size based on tensor dimensions
        let max_size = a.0.len().max(b.0.len());
        let log_size = calculate_log_size(max_size);

        // Generate trace and get result tensor
        let (_trace, c) = gen_add_trace(log_size, &a.0, &b.0);

        let c = vec![Tensor::new(StwoData(Arc::new(c)))];
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

            if is::<luminal::op::Add>(op) {
                *op_ref = Box::new(Add::new(id.index(), Arc::clone(&self.config)))
            } else if is::<luminal::op::Mul>(op) {
                todo!()
            } else if is::<luminal::op::Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else if is::<luminal::op::Function>(op) {
                // Keep the Function operator as is
                continue;
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
