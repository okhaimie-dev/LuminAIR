use luminair_air::{
    components::{
        add::table::{AddColumn, AddTable, AddTableRow},
        max_reduce::table::{MaxReduceColumn, MaxReduceTable, MaxReduceTableRow},
        mul::table::{MulColumn, MulTable, MulTableRow},
        recip::table::{RecipColumn, RecipTable, RecipTableRow},
        sum_reduce::table::{SumReduceColumn, SumReduceTable, SumReduceTableRow},
        log2::table::{Log2Column, Log2Table, Log2TableRow}
    },
    pie::NodeInfo,
};
use luminal::{
    op::{Function as LFunction, *},
    prelude::{petgraph::visit::EdgeRef, *},
};
use num_traits::{identities::Zero, One};
use numerair::{Fixed, SCALE_FACTOR};
use std::{ops::Deref, sync::Arc};
use stwo_prover::core::fields::m31::BaseField;

use crate::{
    data::StwoData,
    utils::{get_buffer_from_tensor, get_index, is},
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

// ================== UNARY ==================

/// Implements element-wise addition for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairRecip {}

impl LuminairRecip {
    /// Creates a new `LuminairRecip` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<RecipColumn, RecipTable> for LuminairRecip {
    /// Processes input tensor, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut RecipTable,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let input = get_buffer_from_tensor(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let output_size = inp[0].1.n_elements().to_usize().unwrap();
        let mut out_data = vec![Fixed::zero(); output_size];

        let node_id: BaseField = node_info.id.into();
        let input_id: BaseField = node_info.inputs[0].id.into();

        for (idx, out) in out_data.iter_mut().enumerate() {
            let input_val = get_index(input, &expr, &mut stack, idx);
            let (out_val, rem_val) = input_val.recip();

            let input_mult = if node_info.inputs[0].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let out_mult = if node_info.output.is_final_output {
                BaseField::zero()
            } else {
                BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
            };

            let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };

            *out = out_val;
            table.add_row(RecipTableRow {
                node_id,
                input_id,
                idx: idx.into(),
                is_last_idx: (is_last_idx).into(),
                next_idx: (idx + 1).into(),
                next_node_id: node_id,
                next_input_id: input_id,
                input: input_val.to_m31(),
                out: out_val.to_m31(),
                rem: rem_val.to_m31(),
                scale: SCALE_FACTOR,
                input_mult,
                out_mult,
            })
        }

        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairRecip {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
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

impl LuminairOperator<AddColumn, AddTable> for LuminairAdd {
    /// Processes two input tensors, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut AddTable,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let (lhs, rhs) = (
            get_buffer_from_tensor(&inp[0].0),
            get_buffer_from_tensor(&inp[1].0),
        );
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let output_size = inp[0].1.n_elements().to_usize().unwrap();
        let mut out_data = vec![Fixed::zero(); output_size];

        let node_id: BaseField = node_info.id.into();
        let lhs_id: BaseField = node_info.inputs[0].id.into();
        let rhs_id: BaseField = node_info.inputs[1].id.into();

        for (idx, out) in out_data.iter_mut().enumerate() {
            let lhs_val = get_index(lhs, &lexpr, &mut stack, idx);
            let rhs_val = get_index(rhs, &rexpr, &mut stack, idx);
            let out_val = lhs_val + rhs_val;
            let lhs_mult = if node_info.inputs[0].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let rhs_mult = if node_info.inputs[1].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let out_mult = if node_info.output.is_final_output {
                BaseField::zero()
            } else {
                BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
            };

            let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };

            *out = out_val;
            table.add_row(AddTableRow {
                node_id,
                lhs_id,
                rhs_id,
                idx: idx.into(),
                is_last_idx: (is_last_idx).into(),
                next_idx: (idx + 1).into(),
                next_node_id: node_id,
                next_lhs_id: lhs_id,
                next_rhs_id: rhs_id,
                lhs: lhs_val.to_m31(),
                rhs: rhs_val.to_m31(),
                out: out_val.to_m31(),
                lhs_mult,
                rhs_mult,
                out_mult,
            })
        }

        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairAdd {
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

impl LuminairOperator<MulColumn, MulTable> for LuminairMul {
    /// Processes two input tensors, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut MulTable,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let (lhs, rhs) = (
            get_buffer_from_tensor(&inp[0].0),
            get_buffer_from_tensor(&inp[1].0),
        );
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let output_size = inp[0].1.n_elements().to_usize().unwrap();
        let mut out_data = vec![Fixed::zero(); output_size];

        let node_id: BaseField = node_info.id.into();
        let lhs_id: BaseField = node_info.inputs[0].id.into();
        let rhs_id: BaseField = node_info.inputs[1].id.into();

        for (idx, out) in out_data.iter_mut().enumerate() {
            let lhs_val = get_index(lhs, &lexpr, &mut stack, idx);
            let rhs_val = get_index(rhs, &rexpr, &mut stack, idx);
            let (out_val, rem_val) = lhs_val * rhs_val;
            let lhs_mult = if node_info.inputs[0].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let rhs_mult = if node_info.inputs[1].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let out_mult = if node_info.output.is_final_output {
                BaseField::zero()
            } else {
                BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
            };

            let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };

            *out = out_val;
            table.add_row(MulTableRow {
                node_id,
                lhs_id,
                rhs_id,
                idx: idx.into(),
                is_last_idx: (is_last_idx).into(),
                next_idx: (idx + 1).into(),
                next_node_id: node_id,
                next_lhs_id: lhs_id,
                next_rhs_id: rhs_id,
                lhs: lhs_val.to_m31(),
                rhs: rhs_val.to_m31(),
                out: out_val.to_m31(),
                rem: rem_val.to_m31(),
                lhs_mult,
                rhs_mult,
                out_mult,
            })
        }

        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairMul {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

// ================== REDUCE ==================

/// Implements SumReduce for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairSumReduce(pub usize);

impl LuminairSumReduce {
    /// Creates a new `LuminairSumReduce` instance.
    pub fn new(value: usize) -> Self {
        Self(value)
    }
}

impl LuminairOperator<SumReduceColumn, SumReduceTable> for LuminairSumReduce {
    /// Processes input tensor, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut SumReduceTable,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];

        let output_size = front_size * back_size;
        let mut out_data = vec![Fixed::zero(); output_size];
        let input = get_buffer_from_tensor(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack: Vec<i64> = vec![];

        let node_id: BaseField = node_info.id.into();
        let input_id: BaseField = node_info.inputs[0].id.into();

        for i in 0..front_size {
            for j in 0..back_size {
                let mut acc = Fixed::zero(); // Initialize accumulator for each (i, j)
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    let input_val = get_index(input, &expr, &mut stack, orig_index);
                    let next_acc = acc + input_val; // Compute next accumulator

                    // Set out_data only in the last reduction step
                    let (out_val, is_last_step) = if k == dim_size - 1 {
                        out_data[i * back_size + j] = next_acc;
                        (next_acc, BaseField::one())
                    } else {
                        (Fixed::zero(), BaseField::zero()) // Placeholder for incomplete reductions
                    };

                    let input_mult = if node_info.inputs[0].is_initializer {
                        BaseField::zero()
                    } else {
                        -BaseField::one()
                    };
                    let out_mult = if node_info.output.is_final_output {
                        BaseField::zero()
                    } else {
                        BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
                    };
                    let idx = i * back_size + j; // Index for out_data

                    let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };
                    // Add row to the trace table with acc and next_acc
                    table.add_row(SumReduceTableRow {
                        node_id,
                        input_id,
                        idx: idx.into(),
                        is_last_idx: (is_last_idx).into(),
                        next_node_id: node_id,
                        next_input_id: input_id,
                        next_idx: (idx + 1).into(),
                        input: input_val.to_m31(),
                        out: out_val.to_m31(),
                        acc: acc.to_m31(),
                        next_acc: next_acc.to_m31(),
                        is_last_step,
                        input_mult,
                        out_mult,
                    });
                    acc = next_acc;
                }
            }
        }
        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairSumReduce {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairMaxReduce(pub usize);

impl LuminairMaxReduce {
    /// Creates a new `LuminairMaxReduce` instance.
    pub fn new(value: usize) -> Self {
        Self(value)
    }
}

impl LuminairOperator<MaxReduceColumn, MaxReduceTable> for LuminairMaxReduce {
    /// Processes input tensor, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut MaxReduceTable,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];

        let output_size = front_size * back_size;
        let mut out_data = vec![Fixed::zero(); output_size];
        let input = get_buffer_from_tensor(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack: Vec<i64> = vec![];

        let node_id: BaseField = node_info.id.into();
        let input_id: BaseField = node_info.inputs[0].id.into();

        for i in 0..front_size {
            for j in 0..back_size {
                // Initialize with the first element instead of negative infinity
                let orig_first_index = i * dim_size * back_size + 0 * back_size + j;
                let mut max_val = get_index(input, &expr, &mut stack, orig_first_index);

                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    let input_val = get_index(input, &expr, &mut stack, orig_index);

                    // Determine if this value is the new max
                    let is_max = if input_val.to_f64() > max_val.to_f64() {
                        BaseField::one()
                    } else {
                        BaseField::zero()
                    };

                    // Update max_val if needed
                    let next_max_val = if is_max == BaseField::one() {
                        input_val
                    } else {
                        max_val
                    };

                    // Set out_data only in the last reduction step
                    let (out_val, is_last_step) = if k == dim_size - 1 {
                        out_data[i * back_size + j] = next_max_val;
                        (next_max_val, BaseField::one())
                    } else {
                        (Fixed::zero(), BaseField::zero()) // Placeholder for incomplete reductions
                    };

                    let input_mult = if node_info.inputs[0].is_initializer {
                        BaseField::zero()
                    } else {
                        -BaseField::one()
                    };
                    let out_mult = if node_info.output.is_final_output {
                        BaseField::zero()
                    } else {
                        BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
                    };
                    let idx = i * back_size + j; // Index for out_data

                    let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };

                    // Add row to the trace table with max_val and next_max_val
                    table.add_row(MaxReduceTableRow {
                        node_id,
                        input_id,
                        idx: idx.into(),
                        is_last_idx: (is_last_idx).into(),
                        next_node_id: node_id,
                        next_input_id: input_id,
                        next_idx: (idx + 1).into(),
                        input: input_val.to_m31(),
                        out: out_val.to_m31(),
                        max_val: max_val.to_m31(),
                        next_max_val: next_max_val.to_m31(),
                        is_last_step,
                        is_max,
                        input_mult,
                        out_mult,
                    });

                    max_val = next_max_val;
                }
            }
        }
        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairMaxReduce {
    /// This method is not used as `process_trace` handles all computation for this operator.
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        unimplemented!()
    }
}

/// Implements element-wise log2 for LuminAIR.
#[derive(Debug, Clone, Default, PartialEq)]
struct LuminairLog2 {}

impl LuminairLog2 {
    /// Creates a new `LuminairLog2` instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl LuminairOperator<Log2Column, Log2Table> for LuminairLog2 {
    /// Processes input tensor, generating a trace, claim, and output tensor.
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        table: &mut Log2Table,
        node_info: &NodeInfo,
    ) -> Vec<Tensor> {
        let input = get_buffer_from_tensor(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());

        let mut stack: Vec<i64> = vec![];
        let output_size = inp[0].1.n_elements().to_usize().unwrap();
        let mut out_data = vec![Fixed::zero(); output_size];

        let node_id: BaseField = node_info.id.into();
        let input_id: BaseField = node_info.inputs[0].id.into();

        for (idx, out) in out_data.iter_mut().enumerate() {
            let input_val = get_index(input, &expr, &mut stack, idx);
            let (out_val, rem_val) = input_val.log2();

            let input_mult = if node_info.inputs[0].is_initializer {
                BaseField::zero()
            } else {
                -BaseField::one()
            };
            let out_mult = if node_info.output.is_final_output {
                BaseField::zero()
            } else {
                BaseField::one() * BaseField::from_u32_unchecked(node_info.num_consumers)
            };

            let is_last_idx: u32 = if idx == (output_size - 1) { 1 } else { 0 };

            *out = out_val;
            table.add_row(Log2TableRow {
                node_id,
                input_id,
                idx: idx.into(),
                is_last_idx: (is_last_idx).into(),
                next_idx: (idx + 1).into(),
                next_node_id: node_id,
                next_input_id: input_id,
                input_val: input_val.to_m31(),
                output_val: out_val.to_m31(),
                pow2_result: (2u32.pow(out_val.to_u32().unwrap())).into(), // Calculate 2^out_val
                input_mult,
                output_mult: out_mult,
            })
        }

        vec![Tensor::new(StwoData(Arc::new(out_data)))]
    }
}

impl Operator for LuminairLog2 {
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
            } else if is::<luminal::op::Mul>(op) {
                *op_ref = LuminairMul::new().into_operator()
            } else if is::<luminal::op::SumReduce>(op) {
                let dim_index =
                    if let Some(sum_reduce) = op_ref.deref().as_any().downcast_ref::<SumReduce>() {
                        sum_reduce.0 // Access the usize field (the 0 in SumReduce(0))
                    } else {
                        0
                    };
                *op_ref = LuminairSumReduce::new(dim_index).into_operator()
            } else if is::<luminal::op::MaxReduce>(op) {
                let dim_index =
                    if let Some(max_reduce) = op_ref.deref().as_any().downcast_ref::<MaxReduce>() {
                        max_reduce.0 // Access the usize field
                    } else {
                        0
                    };
                *op_ref = LuminairMaxReduce::new(dim_index).into_operator()
            } else if is::<luminal::op::Recip>(op) {
                *op_ref = LuminairRecip::new().into_operator()
            } else if is::<luminal::op::Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            }
        }
    }
}
