use crate::op::{
    prim::{CopyFromStwo, CopyToStwo, LuminairConstant},
    HasProcessTrace,
};
use luminair_air::{
    components::{
        add::table::{AddColumn, AddTable},
        ClaimType, LuminairComponents, TraceError,
    },
    pie::{ExecutionResources, InputInfo, LuminairPie, NodeInfo, OpCounter, OutputInfo, Trace},
    serde::SerializableTrace,
    utils::get_is_first_log_sizes,
    LuminairClaim, LuminairProof,
};
use luminal::{
    op::*,
    prelude::{petgraph::visit::EdgeRef, *},
};
use stwo_prover::{
    constraint_framework::{
        preprocessed_columns::IsFirst, ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
    },
    core::{
        backend::simd::SimdBackend,
        channel::Blake2sChannel,
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::circle::{CanonicCoset, PolyOps},
        prover::{self, verify, ProvingError, VerificationError},
        vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher},
    },
};
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum LuminairError {
    #[error(transparent)]
    StwoVerifierError(#[from] VerificationError),

    #[error("{0} lookup values do not match.")]
    InvalidLookup(String),
}

/// Trait defining the core functionality of a LuminAIR computation graph.
///
/// Provides methods to generate execution traces, retrieve outputs, and handle proof
/// generation and verification using Stwo.
pub trait LuminairGraph {
    /// Generates an execution trace for the graph’s computation.
    fn gen_trace(&mut self) -> Result<LuminairPie, TraceError>;

    /// Generates a proof of the graph’s execution using the provided trace.
    fn prove(
        &mut self,
        pie: LuminairPie,
    ) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError>;

    /// Verifies a proof to ensure integrity of graph’s computation.
    fn verify(&self, proof: LuminairProof<Blake2sMerkleHasher>) -> Result<(), LuminairError>;
}

impl LuminairGraph for Graph {
    fn gen_trace(&mut self) -> Result<LuminairPie, TraceError> {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }

        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        // Initialize trace collectors for different operators
        let mut traces = Vec::new();
        // Initializes operator counter
        let mut op_counter = OpCounter::default();

        // Initilializes table for each operator
        let mut add_table = AddTable::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Gather input source information
            let input_info = src_ids
                .iter()
                .map(|(id, _, _)| {
                    let node_weight = self.node_weight(*id).unwrap();
                    let node_is_function = node_weight.as_any().is::<Function>();
                    let node_is_constant = node_weight.as_any().is::<LuminairConstant>()
                        || node_weight.as_any().is::<luminal::op::Constant>();
                    let node_is_copy_to = node_weight.as_any().is::<CopyToStwo>();

                    // Check if this is a CopyToStwo that wraps a Function node or a Constant
                    let is_copy_of_initializer = if node_is_copy_to {
                        self.get_sources(*id).iter().any(|(src_id, _, _)| {
                            let src_weight = self.node_weight(*src_id).unwrap();
                            src_weight.as_any().is::<Function>()
                                || src_weight.as_any().is::<LuminairConstant>()
                                || src_weight.as_any().is::<luminal::op::Constant>()
                        })
                    } else {
                        false
                    };

                    InputInfo {
                        is_initializer: node_is_function
                            || node_is_constant
                            || is_copy_of_initializer,
                    }
                })
                .collect::<Vec<_>>();

            // Get output source information - check if this node is a final output
            // or if it feeds into a CopyFromStwo that's a final output
            let is_direct_output = self.to_retrieve.contains_key(&node);
            let is_output_via_copy = self
                .graph
                .edges_directed(*node, petgraph::Direction::Outgoing)
                .any(|e| {
                    let target = e.target();
                    self.to_retrieve.contains_key(&target)
                        && self
                            .node_weight(target)
                            .unwrap()
                            .as_any()
                            .is::<CopyFromStwo>()
                });

            let output_info = OutputInfo {
                is_final_output: is_direct_output || is_output_via_copy,
            };

            let node_info = NodeInfo {
                inputs: input_info,
                output: output_info,
                num_consumers: *consumers.get(&(*node, 0)).unwrap_or(&0),
            };

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Get operator and try to use process_trace if available
            let node_op = &mut *self.graph.node_weight_mut(*node).unwrap();

            let tensors =
                if <Box<dyn Operator> as HasProcessTrace<AddColumn, AddTable>>::has_process_trace(
                    node_op,
                ) {
                    let tensors = <Box<dyn Operator> as HasProcessTrace<
                        AddColumn,
                        AddTable,
                    >>::call_process_trace(
                        node_op, srcs, &mut add_table
                    )
                    .unwrap();
                    *op_counter.add.get_or_insert(0) += 1;

                    tensors
                } else {
                    // Handle other operators or fallback
                    node_op.process(srcs)
                };

            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }

        self.reset();

        // Convert tables to traces
        let (add_trace, add_claim) = add_table.trace_evaluation()?;
        let max_log_size = add_table.table.len() as u32;

        traces.push(Trace::new(
            SerializableTrace::from(&add_trace),
            ClaimType::Add(add_claim),
        ));

        Ok(LuminairPie {
            traces,
            execution_resources: ExecutionResources {
                op_counter,
                max_log_size,
            },
        })
    }

    fn prove(
        &mut self,
        pie: LuminairPie,
    ) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError> {
   todo!()
    }

    fn verify(
        &self,
        LuminairProof {
            claim,
            proof,
            execution_resources,
        }: LuminairProof<Blake2sMerkleHasher>,
    ) -> Result<(), LuminairError> {
      todo!()
    }
}
