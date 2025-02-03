use luminair_air::{
    components::{add::table::AddColumn, ClaimType, LuminairComponents},
    pie::{LuminairPie, Trace},
    serde::SerializableTrace,
    LuminairClaim, LuminairProof, IS_FIRST_LOG_SIZES, LOG_MAX_ROWS,
};
use luminal::prelude::*;
use stwo_prover::{
    constraint_framework::{
        preprocessed_columns::gen_is_first, ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
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

use crate::{
    data::{OutputConverter, StwoData},
    op::HasProcessTrace,
};

/// Struct to hold input source information
///
/// is_initial is true if the input is coming from an initial input.
#[derive(Debug, Clone)]
pub(crate) struct InputSourceInfo {
    is_initial: bool,
}

pub trait LuminairGraph {
    fn gen_trace(&mut self) -> LuminairPie;
    fn get_final_output(&mut self, id: NodeIndex) -> Vec<f32>;
    fn prove(
        &mut self,
        pie: LuminairPie,
    ) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError>;
    fn verify(&self, proof: LuminairProof<Blake2sMerkleHasher>) -> Result<(), VerificationError>;
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
        let mut traces = Vec::new();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Gather input source information
            let input_sources = src_ids
                .iter()
                .map(|(id, _, _)| InputSourceInfo {
                    is_initial: self.node_weight(*id).unwrap().as_any().is::<Function>(),
                })
                .collect::<Vec<_>>();

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Get operator and try to use process_trace if available
            let node_op = &mut *self.graph.node_weight_mut(*node).unwrap();

            let tensors =
                if <Box<dyn Operator> as HasProcessTrace<AddColumn>>::has_process_trace(node_op) {
                    let (trace, claim, tensors) = <Box<dyn Operator> as HasProcessTrace<
                        AddColumn,
                    >>::call_process_trace(
                        node_op, srcs, input_sources
                    )
                    .unwrap();

                    traces.push(Trace {
                        eval: SerializableTrace::from(&trace),
                        claim: ClaimType::Add(claim),
                    });

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

        LuminairPie { traces }
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

    fn prove(
        &mut self,
        pie: LuminairPie,
    ) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError> {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }

        // ┌──────────────────────────┐
        // │     Protocol Setup       │
        // └──────────────────────────┘

        tracing::info!("Protocol Setup");
        let config = PcsConfig::default();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_MAX_ROWS + config.fri_config.log_blowup_factor + 2)
                .circle_domain()
                .half_coset,
        );
        let channel = &mut Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

        // ┌───────────────────────────────────────────────┐
        // │   Interaction Phase 0 - Preprocessed Trace    │
        // └───────────────────────────────────────────────┘

        tracing::info!("Preprocessed Trace");
        // Generate all preprocessed columns
        let mut tree_builder = commitment_scheme.tree_builder();

        tree_builder.extend_evals(
            IS_FIRST_LOG_SIZES
                .iter()
                .copied()
                .map(gen_is_first::<SimdBackend>),
        );

        // Commit the preprocessed trace
        tree_builder.commit(channel);

        // ┌───────────────────────────────────────┐
        // │        Interaction Phase 1 & 2        │
        // └───────────────────────────────────────┘
        let mut tree_builder_1 = commitment_scheme.tree_builder();
        let mut claim_1 = LuminairClaim::init();

        // Pre-allocate for expected number of traces
        claim_1.add.reserve(pie.traces.len());

        for trace in pie.traces.into_iter() {
            // Consume the traces
            // ┌───────────────────────────────────────┐
            // │    Interaction Phase 1 - Main Trace   │
            // └───────────────────────────────────────┘

            match trace.claim {
                ClaimType::Add(claim) => {
                    // Add the components' trace evaluation to the commit tree.
                    tree_builder_1.extend_evals(trace.eval.to_trace());
                    claim_1.add.push(claim);
                }
            }
        }

        // Mix the claim into the Fiat-Shamir channel.
        claim_1.mix_into(channel);
        // Commit the main trace.
        tree_builder_1.commit(channel);

        // ┌──────────────────────────┐
        // │     Proof Generation     │
        // └──────────────────────────┘
        tracing::info!("Proof Generation");
        let component_builder = LuminairComponents::new(&claim_1);
        let components = component_builder.provers();
        let proof = prover::prove::<SimdBackend, _>(&components, channel, commitment_scheme)?;

        self.reset();

        Ok(LuminairProof {
            claim: claim_1,
            proof,
        })
    }

    fn verify(&self, proof: LuminairProof<Blake2sMerkleHasher>) -> Result<(), VerificationError> {
        // ┌──────────────────────────┐
        // │     Protocol Setup       │
        // └──────────────────────────┘
        let config = PcsConfig::default();
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme_verifier =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let log_sizes = &proof.claim.log_sizes();

        // ┌───────────────────────────────────────────────┐
        // │   Interaction Phase 0 - Preprocessed Trace    │
        // └───────────────────────────────────────────────┘

        commitment_scheme_verifier.commit(
            proof.proof.commitments[PREPROCESSED_TRACE_IDX],
            &log_sizes[PREPROCESSED_TRACE_IDX],
            channel,
        );

        // ┌───────────────────────────────────────┐
        // │    Interaction Phase 1 - Main Trace   │
        // └───────────────────────────────────────┘
        proof.claim.mix_into(channel);
        commitment_scheme_verifier.commit(
            proof.proof.commitments[ORIGINAL_TRACE_IDX],
            &log_sizes[ORIGINAL_TRACE_IDX],
            channel,
        );

        // ┌──────────────────────────┐
        // │    Proof Verification    │
        // └──────────────────────────┘

        let component_builder = LuminairComponents::new(&proof.claim);
        let components = component_builder.components();

        verify(
            &components,
            channel,
            commitment_scheme_verifier,
            proof.proof,
        )
    }
}
