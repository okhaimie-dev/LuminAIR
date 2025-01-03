use luminal::prelude::*;
use std::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use stwo_prover::core::backend::{
    simd::{m31::LOG_N_LANES, SimdBackend},
    Backend,
};

use crate::{
    air::{add::trace::TensorAddTracer, tensor::AirTensor},
    compiler::data::PackedData,
};

use super::data::{FromTensorField, TensorData};

#[derive(Debug)]
pub struct PrimitiveCompiler {}

////////////// BINARY //////////////

#[derive(Debug, Clone, PartialEq)]
pub struct StwoAdd<B: Backend + 'static, D> {
    _phantom: PhantomData<(B, D)>,
}

impl<B, D> StwoAdd<B, D>
where
    B: Backend + 'static + TensorAddTracer<D::Field>,
    D: TensorData + FromTensorField<D::Field>,
{
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<B, D> Operator for StwoAdd<B, D>
where
    B: Backend + 'static + TensorAddTracer<D::Field>,
    D: TensorData + FromTensorField<D::Field>,
{
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp.len() != 2 {
            panic!("Add operator requires exactly two input tensors.");
        }

        // Get references to the data
        let a_data = inp[0]
            .0
            .borrowed()
            .downcast_ref::<D>()
            .expect("Expected correct tensor data type")
            .as_slice();
        let b_data = inp[1]
            .0
            .borrowed()
            .downcast_ref::<D>()
            .expect("Expected correct tensor data type")
            .as_slice();

        // Create AirTensors
        let a = AirTensor {
            data: a_data.to_vec(), // TODO(@raphaelDkhn): avoid cloning
            dims: inp[0].1.shape_usize(),
            stride: AirTensor::<D::Field>::compute_stride(&inp[0].1.shape_usize()),
        };
        let b = AirTensor {
            data: b_data.to_vec(), // TODO(@raphaelDkhn): avoid cloning
            dims: inp[1].1.shape_usize(),
            stride: AirTensor::<D::Field>::compute_stride(&inp[1].1.shape_usize()),
        };

        // Calculate required log_size based on tensor dimensions
        let max_size = a.size().max(b.size());
        let required_log_size = ((max_size + (1 << LOG_N_LANES) - 1) >> LOG_N_LANES)
            .next_power_of_two()
            .trailing_zeros()
            + LOG_N_LANES;

        // Generate trace and get result
        let (_, result) = B::generate_trace(required_log_size, &a, &b);

        // Convert result back to Tensor
        vec![Tensor::new(D::from_tensor_field(result.data))]
    }
}

impl Compiler for PrimitiveCompiler {
    type Output = ();

    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _ids: T) -> Self::Output {
        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        for id in graph.node_indices().collect::<Vec<_>>() {
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Add>(op) {
                // TODO(@raphaelDkhn): make backend and data type configurable
                *op_ref = Box::new(StwoAdd::<SimdBackend, PackedData>::new());
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
