use stwo_prover::{
    constraint_framework::preprocessed_columns::gen_is_first,
    core::{
        backend::simd::SimdBackend,
        channel::Blake2sChannel,
        pcs::{CommitmentSchemeProver, PcsConfig},
        poly::circle::{CanonicCoset, PolyOps},
        vcs::blake2_merkle::Blake2sMerkleChannel,
    },
};

use crate::components::{Claim, TraceEval};

/// `LOG_MAX_ROWS = ilog2(MAX_ROWS)`
///
/// Means that Luminair does not accept programs inducing a component with more than 2^LOG_MAX_ROWS steps
const LOG_MAX_ROWS: u32 = 14;

/// Log sizes of the preprocessed columns
/// used for enforcing boundary constraints.
///
/// Preprocessed columns are generated ahead of time,
/// so at this moment we don't know the log size
/// of the main and interaction traces.
///
/// Therefore, we generate all log sizes that we
/// want to support, so that the verifier can be
/// provided a merkle root it can trust, for a claim
/// of any dynamic size.
///
/// Ideally, we should cover all possible log sizes, between
/// 1 and `LOG_MAX_ROW`
const IS_FIRST_LOG_SIZES: [u32; 12] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4];

pub fn prove_graph(traces: Vec<(TraceEval, Claim)>) {
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
    // │    Interaction Phase 1 - Main Trace   │
    // └───────────────────────────────────────┘

    tracing::info!("Main Trace");
    let mut tree_builder = commitment_scheme.tree_builder();

    // Add the components' trace evaluation to the commit tree.
    for trace in traces {
        tree_builder.extend_evals(trace.0);

        // Mix the claim into the Fiat-Shamir channel.
        trace.1.mix_into(channel);
    }

    // Commit the main trace.
    tree_builder.commit(channel);
}
