use luminair_air::{
    components::{
        add::{self, table::AddColumn},
        log2::{self, table::Log2Column},
        mul::{self, table::MulColumn},
        Claim, TraceEval,
    },
    pie::NodeInfo,
};
use luminal::{
    op::{Function as LFunction, *},
    prelude::{petgraph::visit::EdgeRef, *},
};
use num_traits::identities::Zero;
use numerair::Fixed;
use std::sync::Arc;

use crate::{
    data::StwoData,
    utils::{get_buffer_from_tensor, is},
};

use super::{IntoOperator, LuminairOperator};

// ================== COPY ==================

/// Copy a tensor to Stwo
#[derive(Clone, Debug)]
pub struct CopyToStwo {}
impl CopyToStwo {
    /// Creates a new `CopyToStwo` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl Operator for CopyToStwo {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<StwoData>() {
            // Already in StwoData format, no conversion needed
            return vec![inp.pop().unwrap().0.cloned()];
        }

        // Convert Vec<f32> to StwoData
        let cpu_data = inp[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        vec![Tensor::new(StwoData::from_f32(cpu_data))]
    }
}

/// Copy a tensor from Stwo
#[derive(Clone, Debug)]
pub struct CopyFromStwo {}
impl CopyFromStwo {
    /// Creates a new `CopyFromStwo` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl Operator for CopyFromStwo {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<Vec<f32>>() {
            // Already in Vec<f32> format, no conversion needed
            return vec![inp.pop().unwrap().0.cloned()];
        }

        // Convert StwoData to Vec<f32>
        let data = inp[0].0.borrowed().downcast_ref::<StwoData>().unwrap();
        vec![Tensor::new(data.to_f32())]
    }
}

// ================== CONSTANT ================

/// Implements a constant operator for LuminAIR.
#[derive(Debug, Clone, PartialEq)]
pub struct LuminairConstant {
    pub value: ConstantValue,
}

impl LuminairConstant {
    /// Creates a new LuminairConstant with the given value.
    pub fn new(value: ConstantValue) -> Self {
        Self { value }
    }
}

impl Operator for LuminairConstant {
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Create a new tensor with the constant value
        let value = match &self.value {
            ConstantValue::Float(f) => *f,
            ConstantValue::Expression(_expr) => {
                panic!("Dynamic expressions not yet supported")
            }
        };

        // Create and return a single element with the constant value
        let mut data = Vec::with_capacity(1);
        data.push(Fixed::from_f64(value as f64));
        vec![Tensor::new(StwoData(Arc::new(data)))]
    }
}

// ================== BINARY ==================

/// Implements element-wise addition for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairAdd {}

impl LuminairAdd {
    /// Creates a new `LuminairAdd` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<AddColumn> for LuminairAdd {
    /// Processes two input tensors, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> (TraceEval, Claim<AddColumn>, Vec<Tensor>) {
        let (lhs, rhs) = (
            get_buffer_from_tensor(&inp[0].0),
            get_buffer_from_tensor(&inp[1].0),
        );
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let mut out_data = vec![Fixed::zero(); inp[0].1.n_elements().to_usize().unwrap()];

        // Generate trace and claim
        let (main_trace, claim) = add::table::trace_evaluation(
            &lhs.0,
            &rhs.0,
            &lexpr,
            &rexpr,
            &mut stack,
            &mut out_data,
            node_info,
        );

        (
            main_trace,
            claim,
            vec![Tensor::new(StwoData(Arc::new(out_data)))],
        )
    }
}

impl Operator for LuminairAdd {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

/// Implements element-wise base-2 logarithm for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairLog2 {}

impl LuminairLog2 {
    /// Creates a new `LuminairLog2` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<Log2Column> for LuminairLog2 {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> (TraceEval, Claim<Log2Column>, Vec<Tensor>) {
        let input = get_buffer_from_tensor(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let mut out_data = vec![Fixed::zero(); inp[0].1.n_elements().to_usize().unwrap()];

        // Generate trace and claim
        let (main_trace, claim) =
            log2::table::trace_evaluation(&input.0, &expr, &mut stack, &mut out_data, node_info);

        (
            main_trace,
            claim,
            vec![Tensor::new(StwoData(Arc::new(out_data)))],
        )
    }
}

impl Operator for LuminairLog2 {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

/// Implements element-wise multiplication for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairMul {}

impl LuminairMul {
    /// Creates a new `LuminairMul` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<MulColumn> for LuminairMul {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> (TraceEval, Claim<MulColumn>, Vec<Tensor>) {
        let (lhs, rhs) = (
            get_buffer_from_tensor(&inp[0].0),
            get_buffer_from_tensor(&inp[1].0),
        );
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let mut out_data = vec![Fixed::zero(); inp[0].1.n_elements().to_usize().unwrap()];

        // Generate trace and claim
        let (main_trace, claim) = mul::table::trace_evaluation(
            &lhs.0,
            &rhs.0,
            &lexpr,
            &rexpr,
            &mut stack,
            &mut out_data,
            node_info,
        );

        (
            main_trace,
            claim,
            vec![Tensor::new(StwoData(Arc::new(out_data)))],
        )
    }
}

impl Operator for LuminairMul {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

// ================== COMPILER ==================

/// Compiles primitive operations into provable forms for LuminAIR.
///
/// Replaces standard Luminal operators with LuminAIR-specific implementations
/// that support trace generation.
#[derive(Default)]
pub struct PrimitiveCompiler {}

impl PrimitiveCompiler {
    /// Creates a new `PrimitiveCompiler` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl Compiler for PrimitiveCompiler {
    type Output = ();

    /// Compiles a graph by replacing Luminal operators with LuminAIR equivalents.
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) -> Self::Output {
        // Go through the graph and insert copy ops.
        // Copy Function nodes (data input/output)
        for function_node in graph
            .node_indices()
            .filter(|n| {
                graph.node_weight(*n).unwrap().as_any().is::<Function>()
                    && graph.edges(*n).count() != 0
            })
            .collect::<Vec<_>>()
        {
            // Create CopyToStwo to convert Vec<f32> data to StwoData after function outputs
            let copy_node = graph
                .add_op(CopyToStwo::new())
                .input(function_node, 0, ShapeTracker::new(()))
                .finish();

            // Switch outgoing edges from input to copy_node
            for (edge_id, weight, dest) in graph
                .edges_directed(function_node, petgraph::Direction::Outgoing)
                .map(|e| (e.id(), *e.weight(), e.target()))
                .filter(|(_, _, trg)| *trg != copy_node)
                .collect::<Vec<_>>()
            {
                graph.add_edge(copy_node, dest, weight);
                graph.remove_edge(edge_id);
            }

            // Handle no_delete and to_retrieve for the function node
            if graph.no_delete.remove(&function_node) {
                graph.no_delete.insert(copy_node);
            }
            if let Some(v) = graph.to_retrieve.get(&function_node) {
                graph.to_retrieve.insert(copy_node, *v);
            }

            // Insert copy from Stwo for function inputs
            for (source, edge, edge_weight) in graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(CopyFromStwo::new())
                    .input(source, 0, ShapeTracker::new(()))
                    .finish();
                graph.add_edge(copy_from_node, function_node, edge_weight);
                graph.remove_edge(edge);
            }
        }

        // Add CopyFromStwo for retrieved outputs
        for (output_node, (_, output_shape)) in graph
            .to_retrieve
            .iter()
            .map(|(a, b)| (*a, *b))
            // Filter to non-functions
            .filter(|(n, _)| !graph.node_weight(*n).unwrap().as_any().is::<LFunction>())
            .collect::<Vec<_>>()
        {
            if graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<CopyToStwo>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                graph.no_delete.remove(&output_node);
                graph.no_delete.insert(src);
                let w = graph.to_retrieve.remove(&output_node).unwrap();
                graph.to_retrieve.insert(src, w);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(CopyFromStwo::new())
                    .input(output_node, 0, output_shape)
                    .finish();

                remap(output_node, copy_node, &mut ids, graph);
            }
        }

        // Replace Luminal's ops with LuminAIR ops
        for id in graph.node_indices().collect::<Vec<_>>() {
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if let Some(c) = op_ref.as_any().downcast_ref::<luminal::op::Constant>() {
                *op_ref = Box::new(LuminairConstant::new(c.0.clone()));
            } else if is::<luminal::op::Add>(op) {
                *op_ref = LuminairAdd::new().into_operator()
            } else if is::<luminal::op::Log2>(op) {
                *op_ref = LuminairLog2::new().into_operator()
            } else if is::<luminal::op::Mul>(op) {
                *op_ref = LuminairMul::new().into_operator()
            } else if is::<luminal::op::Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            }
        }
    }
}
