use serde::{Deserialize, Serialize};
use stwo_prover::{
    constraint_framework::{
        preprocessed_columns::gen_is_first, ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
    },
    core::{
        backend::simd::SimdBackend,
        channel::Blake2sChannel,
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::circle::{CanonicCoset, PolyOps},
        prover::{prove, verify, ProvingError, StarkProof, VerificationError},
        vcs::{
            blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher},
            ops::MerkleHasher,
        },
    },
};

use crate::{LuminairClaim, LuminairComponents, LuminairTrace};

/// `LOG_MAX_ROWS = ilog2(MAX_ROWS)`
///
/// Means that Luminair does not accept programs inducing a component with more than 2^LOG_MAX_ROWS steps
const LOG_MAX_ROWS: u32 = 14;

/// The STARK proof of the execution of a given Luminair Graph.
///
/// It includes the proof as well as the claims during the various phases of the proof generation.
#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairProof<H: MerkleHasher> {
    pub claim: LuminairClaim,
    pub proof: StarkProof<H>,
}

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
pub(crate) const IS_FIRST_LOG_SIZES: [u32; 12] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4];

/// Generate a STARK proof of the given Luminair computational graph.
pub fn prover(
    luminair_trace: LuminairTrace,
) -> Result<LuminairProof<Blake2sMerkleHasher>, ProvingError> {
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
    for trace in luminair_trace.traces {
        tree_builder.extend_evals(trace);
    }
    luminair_trace.claims.mix_into(channel);

    // Commit the main trace.
    tree_builder.commit(channel);

    // ┌───────────────────────────────────────────────┐
    // │    Interaction Phase 2 - Interaction Trace    │
    // └───────────────────────────────────────────────┘

    // No interection trace with the components for the moment.

    // ┌──────────────────────────┐
    // │     Proof Generation     │
    // └──────────────────────────┘
    tracing::info!("Proof Generation");
    let component_builder = LuminairComponents::new(&luminair_trace.claims);
    let components = component_builder.provers();
    let proof = prove::<SimdBackend, _>(&components, channel, commitment_scheme)?;

    Ok(LuminairProof {
        claim: luminair_trace.claims,
        proof,
    })
}

/// Verify a given STARK proof of a Luminair computational graph with corresponding claim.
pub fn verifier(
    LuminairProof { claim, proof }: LuminairProof<Blake2sMerkleHasher>,
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

    // No interaction trace for the moment

    // ┌──────────────────────────┐
    // │    Proof Verification    │
    // └──────────────────────────┘
    let component_builder = LuminairComponents::new(&claim);
    let components = component_builder.components();

    verify(&components, channel, commitment_scheme_verifier, proof)
}
