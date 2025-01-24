use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use luminair_air::{
    components::{
        add::trace::{gen_add_trace, AddColumn},
        Claim, TraceEval,
    },
    utils::calculate_log_size,
};
use luminal::prelude::*;

use crate::{data::StwoData, op::IntoOperator};

use super::LuminairOperator;

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
pub struct LuminairAdd {
    node_id: usize,
    config: Arc<Config>,
}

impl LuminairAdd {
    pub fn new(node_id: usize, config: Arc<Config>) -> Self {
        Self { node_id, config }
    }
}

impl LuminairOperator<AddColumn> for LuminairAdd {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> (Vec<Tensor>, Claim<AddColumn>, TraceEval) {
        if inp.len() != 2 {}

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
        let (trace, log_size, c) = gen_add_trace(log_size, &a.0, &b.0);

        let c = vec![Tensor::new(StwoData(Arc::new(c)))];
        (c, log_size, trace)
    }
}

impl Operator for LuminairAdd {
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // We don't need to implement process as we implement process_trace for this op.
        unimplemented!()
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
                *op_ref = LuminairAdd::new(id.index(), Arc::clone(&self.config)).into_operator()
            } else if is::<luminal::op::Mul>(op) {
                todo!()
            } else if is::<luminal::op::Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else if is::<luminal::op::Function>(op) {
                continue;
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
