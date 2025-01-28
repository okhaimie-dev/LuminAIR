use std::collections::HashMap;

use crate::{
    data::{OutputConverter, StwoData},
    op::HasProcessTrace,
};
use luminair_air::{
    components::add::trace::AddColumn,
    pie::{ClaimType, LuminairPie, Phase1, Trace},
    prover::{prover, verifier, LuminairProof},
    serde::SerializableTrace,
};
use luminal::prelude::*;
use stwo_prover::core::{
    prover::{ProvingError, VerificationError},
    vcs::blake2_merkle::Blake2sMerkleHasher,
};

pub trait LuminairGraph {
    fn gen_trace(&mut self) -> LuminairPie;
    fn prove(&self, pie: LuminairPie) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError>;
    fn verify(&self, proof: LuminairProof<Blake2sMerkleHasher>) -> Result<(), VerificationError>;
    fn get_final_output(&mut self, id: NodeIndex) -> Vec<f32>;
}

impl LuminairGraph for Graph {
    /// Execute the graph and generate trace.
    fn gen_trace(&mut self) -> LuminairPie {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        // Initialize trace collectors for different operators
        let mut traces: HashMap<usize, Trace> = HashMap::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Get operator and try to use process_trace if available
            let node_op = &mut *self.graph.node_weight_mut(*node).unwrap();

            let tensors =
                if <Box<dyn Operator> as HasProcessTrace<AddColumn>>::has_process_trace(node_op) {
                    let (tensors, claim, trace) =
                        <Box<dyn Operator> as HasProcessTrace<AddColumn>>::call_process_trace(
                            node_op, srcs,
                        )
                        .unwrap();

                    let phase_1 = Phase1 {
                        trace: SerializableTrace::from(&trace),
                        claim: ClaimType::Add(claim),
                    };

                    traces.insert(
                        node.index(),
                        Trace {
                            phase_1,
                            phase_2: None,
                        },
                    );

                    tensors
                }
                // else if <Box<dyn Operator> as HasProcessTrace<MulColumn>>::has_process_trace(node_op) {
                //     let (tensors, claim, trace) =
                //         <Box<dyn Operator> as HasProcessTrace<MulColumn>>::call_process_trace(node_op, srcs)
                //         .unwrap();
                //     mul_traces.push(trace);
                //     mul_claims.push(claim);
                //     tensors
                // }
                else {
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

        LuminairPie {
            execution_resources: None,
            traces,
        }
    }

    fn prove(&self, pie: LuminairPie) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError> {
        prover(pie)
    }

    fn verify(&self, proof: LuminairProof<Blake2sMerkleHasher>) -> Result<(), VerificationError> {
        verifier(proof)
    }

    fn get_final_output(&mut self, id: NodeIndex) -> Vec<f32> {
        // Get the shape from the graph edges
        let output_size = if let Some((_, shape)) = self.to_retrieve.get(&id) {
            shape
                .n_elements()
                .to_usize()
                .expect("Failed to get tensor size")
        } else {
            // Fallback to checking graph edges if not in to_retrieve
            self.graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .find_map(|e| e.weight().as_data())
                .map(|(_, _, shape)| {
                    shape
                        .n_elements()
                        .to_usize()
                        .expect("Failed to get tensor size")
                })
                .expect("Could not determine tensor shape")
        };

        if let Some(tensor) = self.tensors.remove(&(id, 0)) {
            if let Some(data) = tensor.downcast_ref::<StwoData>() {
                let converter = OutputConverter::new(data.clone(), output_size);
                return converter.to_f32();
            }
        }
        panic!("No StwoData found for final output conversion");
    }
}
