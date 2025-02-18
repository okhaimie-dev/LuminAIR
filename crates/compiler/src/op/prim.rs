use std::sync::Arc;

use luminair_air::{
    components::{
        add::table::{trace_evaluation, AddColumn},
        Claim, TraceEval,
    },
    utils::calculate_log_size,
};
use luminal::prelude::*;

use crate::{data::StwoData, utils::is};

use super::{IntoOperator, LuminairOperator};

#[derive(Default)]
pub struct PrimitiveCompiler {}

impl PrimitiveCompiler {
    pub fn new() -> Self {
        Self {}
    }
}

// ====== BINARY ======
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairAdd {}

impl LuminairAdd {
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<AddColumn> for LuminairAdd {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> (TraceEval, Claim<AddColumn>, Vec<Tensor>) {
        // Get data
        let (lhs_tensor, _) = &inp[0];
        let (rhs_tensor, _) = &inp[1];

        let get_data = |tensor: &InputTensor| {
            if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
                StwoData::from_f32(data)
            } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
                StwoData(Arc::clone(&data.0))
            } else {
                panic!("Unsupported input type for Add");
            }
        };

        let lhs = get_data(lhs_tensor);
        let rhs = get_data(rhs_tensor);

        // Calculate required trace size based on tensor dimensions
        let max_size = lhs.0.len().max(rhs.0.len());
        let log_size = calculate_log_size(max_size);

        // Generate trace and get result tensor
        let (main_trace, claim, output) = trace_evaluation(log_size, &lhs.0, &rhs.0);

        (
            main_trace,
            claim,
            vec![Tensor::new(StwoData(Arc::new(output)))],
        )
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
        for id in graph.node_indices().collect::<Vec<_>>() {
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<luminal::op::Add>(op) {
                *op_ref = LuminairAdd::new().into_operator()
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
