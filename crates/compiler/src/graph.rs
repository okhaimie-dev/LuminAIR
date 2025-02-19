use luminair_air::{
    components::{
        add::table::{interaction_trace_evaluation, AddColumn},
        ClaimType, LuminairComponents, LuminairInteractionElements,
    },
    pie::{ExecutionResources, IOInfo, InputInfo, LuminairPie, OpCounter, OutputInfo, Trace},
    serde::SerializableTrace,
    utils::lookup_sum_valid,
    LuminairClaim, LuminairInteractionClaim, LuminairProof, IS_FIRST_LOG_SIZES, LOG_MAX_ROWS,
};
use luminal::prelude::*;
use stwo_prover::{
    constraint_framework::{
        preprocessed_columns::gen_is_first, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
        PREPROCESSED_TRACE_IDX,
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
        // Initializes operator counter
        let mut op_counter = OpCounter::default();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Gather input source information
            let input_info = src_ids
                .iter()
                .map(|(id, _, _)| InputInfo {
                    is_initializer: self.node_weight(*id).unwrap().as_any().is::<Function>(),
                })
                .collect::<Vec<_>>();

            // Get output source information
            let output_info = OutputInfo {
                is_final_output: self.to_retrieve.contains_key(&node),
            };

            let io_info = IOInfo {
                inputs: input_info,
                output: output_info,
            };

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Get operator and try to use process_trace if available
            let node_op = &mut *self.graph.node_weight_mut(*node).unwrap();

            let tensors =
                if <Box<dyn Operator> as HasProcessTrace<AddColumn>>::has_process_trace(node_op) {
                    let (trace, claim, tensors) =
                        <Box<dyn Operator> as HasProcessTrace<AddColumn>>::call_process_trace(
                            node_op, srcs, &io_info,
                        )
                        .unwrap();

                    traces.push(Trace {
                        eval: SerializableTrace::from(&trace),
                        claim: ClaimType::Add(claim),
                        io_info,
                    });
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

        LuminairPie {
            traces,
            execution_resources: ExecutionResources { op_counter },
        }
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
        mut pie: LuminairPie,
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

        let mut interactions = Vec::new();
        let interaction_elements =
            LuminairInteractionElements::draw(channel, &pie.execution_resources.op_counter);

        let mut interaction_elements_clone = interaction_elements.clone();

        // Pre-allocate for expected number of traces
        claim_1
            .add
            .reserve(*pie.execution_resources.op_counter.add.get_or_insert(0));

        for trace in pie.traces.into_iter() {
            match trace.claim {
                ClaimType::Add(claim) => {
                    let io_info = trace.io_info;
                    let trace = trace.eval.to_trace();
                    // Add the components' trace evaluation to the commit tree.
                    tree_builder_1.extend_evals(trace.clone());
                    claim_1.add.push(claim);

                    let int_el = interaction_elements_clone
                        .add_lookup_elements
                        .pop_front()
                        .unwrap();

                    interactions
                        .push(interaction_trace_evaluation(&trace, &int_el, &io_info).unwrap());
                }
            }
        }

        // Mix the claim into the Fiat-Shamir channel.
        claim_1.mix_into(channel);
        // Commit the main trace.
        tree_builder_1.commit(channel);

        let mut tree_builder_2 = commitment_scheme.tree_builder();
        let mut claim_2 = LuminairInteractionClaim::init();

        for (trace_eval, claim) in interactions {
            tree_builder_2.extend_evals(trace_eval);
            claim_2.add.push(claim);
        }

        // Mix the interaction claim into the Fiat-Shamir channel.
        claim_2.mix_into(channel);
        // Commit the interaction trace.
        tree_builder_2.commit(channel);

        // ┌──────────────────────────┐
        // │     Proof Generation     │
        // └──────────────────────────┘
        tracing::info!("Proof Generation");
        let component_builder = LuminairComponents::new(&claim_1, &interaction_elements, &claim_2);
        let components = component_builder.provers();
        let proof = prover::prove::<SimdBackend, _>(&components, channel, commitment_scheme)?;

        self.reset();

        Ok(LuminairProof {
            claim: claim_1,
            interaction_claim: claim_2,
            proof,
            execution_resources: pie.execution_resources,
        })
    }

    fn verify(
        &self,
        LuminairProof {
            claim,
            interaction_claim,
            proof,
            execution_resources,
        }: LuminairProof<Blake2sMerkleHasher>,
    ) -> Result<(), VerificationError> {
        // ┌──────────────────────────┐
        // │     Protocol Setup       │
        // └──────────────────────────┘
        let config = PcsConfig::default();
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme_verifier =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        let log_sizes = &claim.log_sizes();

        // ┌───────────────────────────────────────────────┐
        // │   Interaction Phase 0 - Preprocessed Trace    │
        // └───────────────────────────────────────────────┘

        commitment_scheme_verifier.commit(
            proof.commitments[PREPROCESSED_TRACE_IDX],
            &log_sizes[PREPROCESSED_TRACE_IDX],
            channel,
        );

        // ┌───────────────────────────────────────┐
        // │    Interaction Phase 1 - Main Trace   │
        // └───────────────────────────────────────┘
        claim.mix_into(channel);
        commitment_scheme_verifier.commit(
            proof.commitments[ORIGINAL_TRACE_IDX],
            &log_sizes[ORIGINAL_TRACE_IDX],
            channel,
        );

        // ┌───────────────────────────────────────────────┐
        // │    Interaction Phase 2 - Interaction Trace    │
        // └───────────────────────────────────────────────┘

        let interaction_elements =
            LuminairInteractionElements::draw(channel, &execution_resources.op_counter);

        // Check that the lookup sum is valid, otherwise throw
        if !lookup_sum_valid(&interaction_claim) {
            return Err(VerificationError::InvalidLookup(
                "Invalid LogUp sum".to_string(),
            ));
        };

        interaction_claim.mix_into(channel);
        commitment_scheme_verifier.commit(
            proof.commitments[INTERACTION_TRACE_IDX],
            &log_sizes[INTERACTION_TRACE_IDX],
            channel,
        );

        // ┌──────────────────────────┐
        // │    Proof Verification    │
        // └──────────────────────────┘

        let component_builder =
            LuminairComponents::new(&claim, &interaction_elements, &interaction_claim);
        let components = component_builder.components();

        verify(&components, channel, commitment_scheme_verifier, proof)
    }
}
