use std::{
    any::{Any, TypeId},
    path::PathBuf,
    sync::Arc,
};

use luminal::prelude::*;

use crate::data::StwoData;

#[derive(Default)]
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

        let (a_tensor, _a_shape) = &inp[0];
        let (b_tensor, _b_shape) = &inp[1];

        let get_data = |tensor: &InputTensor| {
            if let Some(data) = tensor.borrowed().downcast_ref::<Vec<f32>>() {
                StwoData::from_f32(data)
            } else if let Some(data) = tensor.borrowed().downcast_ref::<StwoData>() {
                StwoData(Arc::clone(&data.0))
            } else {
                panic!("Unsupported input type for StwoAdd");
            }
        };

        let _a_data = get_data(a_tensor);
        let _b_data = get_data(b_tensor);

        todo!()
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
            } else if is::<Mul>(op) {
                todo!()
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
