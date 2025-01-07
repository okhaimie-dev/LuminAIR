use std::{
    any::{Any, TypeId},
    path::PathBuf,
    sync::Arc,
};

use luminal::prelude::*;

use crate::data::StwoData;
use luminair_air::{
    ops::add::simd::trace::generate_trace, serde::SerializableTrace, tensor::AirTensor,
    utils::calculate_log_size,
};

#[derive(Debug, Default)]
pub struct PrimitiveCompiler {
    config: Arc<Config>,
}
impl PrimitiveCompiler {
    pub fn new(config: Config) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Config {
    pub trace_registry: Option<PathBuf>,
}

// ====== BINARY ======
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StwoAdd {
    node_id: usize,
    config: Arc<Config>,
}

impl StwoAdd {
    pub fn new(node_id: usize, config: Arc<Config>) -> Self {
        Self { node_id, config }
    }
}

impl Operator for StwoAdd {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp.len() != 2 {
            panic!("Add operator requires exactly two input tensors.");
        }

        let (a_tensor, a_shape) = &inp[0];
        let (b_tensor, b_shape) = &inp[1];

        let get_data = |tensor: &InputTensor| {
            if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
                StwoData::from_f32(data)
            } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
                StwoData(Arc::clone(&data.0))
            } else {
                panic!("Unsupported input type for StwoAdd");
            }
        };

        let a_data = get_data(a_tensor);
        let b_data = get_data(b_tensor);

        // Create AirTensors
        let a = AirTensor::new(a_data.as_slice(), a_shape.shape_usize());
        let b = AirTensor::new(b_data.as_slice(), b_shape.shape_usize());

        // Calculate required trace size based on tensor dimensions
        let max_size = a.size().max(b.size());
        let required_log_size = calculate_log_size(max_size);

        // Generate trace and get result tensor
        let (trace, c) = generate_trace(required_log_size, &a, &b);

        // Save trace if trace_registry is present
        if let Some(trace_registry) = &self.config.trace_registry {

            let file_path = trace_registry.join(format!("{}_add.bin", self.node_id));

            let serializable = SerializableTrace::from(&trace);
            if let Err(err) = serializable.save(file_path) {
                eprintln!("Failed to save trace: {:?}", err);
            }
        }

        let c = vec![Tensor::new(StwoData(Arc::new(c.into_data_vec())))];
        c
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
                *op_ref = Box::new(StwoAdd::new(id.index(), Arc::clone(&self.config)))
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else if is::<Function>(op) {
                // Keep the Function operator as is
                continue;
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
