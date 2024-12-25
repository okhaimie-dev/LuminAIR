use std::sync::Arc;

use crate::circuits::Tensor;
use parking_lot::Mutex;
use rayon::iter::{ParallelBridge, ParallelIterator};
use stwo_prover::core::backend::Column;
use stwo_prover::core::{
    backend::{
        simd::{m31::LOG_N_LANES, SimdBackend},
        Col,
    },
    fields::m31::BaseField,
    poly::{
        circle::{CanonicCoset, CircleEvaluation},
        BitReversedOrder,
    },
    ColumnVec,
};

pub fn generate_trace(
    log_size: u32,
    a: &Tensor,
    b: &Tensor,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    Tensor,
) {
    assert!(a.is_broadcastable_with(b), "Tensors must be broadcastable");

    // Calculate required trace size
    let max_size = a.size().max(b.size());
    assert!(log_size >= LOG_N_LANES);

    // Initialize trace columns
    let trace_size = 1 << log_size;
    let mut trace = Vec::with_capacity(3);
    for _ in 0..3 {
        trace.push(Col::<SimdBackend, BaseField>::zeros(trace_size));
    }

    let trace = Arc::new(
        trace
            .into_iter()
            .map(|col| Mutex::new(col))
            .collect::<Vec<_>>(),
    );

    let c_data = Arc::new(Mutex::new(Vec::with_capacity(
        (max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES,
    )));

    // Calculate number of SIMD-packed rows needed for each tensor
    let n_rows = 1 << (log_size - LOG_N_LANES);
    let a_packed_size = (a.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;
    let b_packed_size = (b.size() + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES;

    // Fill trace with tensor data
    // Process chunks in parallel
    const CHUNK_SIZE: usize = 256;
    (0..n_rows)
        .into_iter()
        .step_by(CHUNK_SIZE)
        .par_bridge()
        .for_each(|chunk| {
            let end = (chunk + CHUNK_SIZE).min(n_rows);
            let mut local_c_data = Vec::with_capacity(CHUNK_SIZE);

            for vec_row in chunk..end {
                if vec_row < max_size {
                    let a_idx = vec_row % a_packed_size;
                    let b_idx = vec_row % b_packed_size;

                    let sum = a.data()[a_idx] + b.data()[b_idx];

                    trace[0].lock().data[vec_row] = a.data()[a_idx];
                    trace[1].lock().data[vec_row] = b.data()[b_idx];
                    trace[2].lock().data[vec_row] = sum;

                    local_c_data.push(sum);
                }
            }

            // Append local results to global c_data
            let mut c_data = c_data.lock();
            c_data.extend(local_c_data);
        });

    let trace = Arc::try_unwrap(trace)
        .unwrap()
        .into_iter()
        .map(|mutex| mutex.into_inner())
        .collect::<Vec<_>>();

    let c_data = Arc::try_unwrap(c_data).unwrap().into_inner();

    // Create output tensor C
    let c = Tensor {
        data: c_data,
        dims: if a.size() > b.size() {
            a.dims().to_vec()
        } else {
            b.dims().to_vec()
        },
        stride: Tensor::compute_stride(if a.size() > b.size() {
            a.dims()
        } else {
            b.dims()
        }),
    };

    let domain = CanonicCoset::new(log_size).circle_domain();

    (
        trace
            .into_iter()
            .map(|eval| CircleEvaluation::new(domain, eval))
            .collect(),
        c,
    )
}
