use std::any::{Any, TypeId};

use luminal::prelude::*;

#[derive(Debug)]
pub struct PrimitiveCompiler {}

// ====== BINARY ======
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StwoAdd;
impl Operator for StwoAdd {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp.len() != 2 {
            panic!("Add operator requires exactly two input tensors.");
        }

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
                *op_ref = Box::new(StwoAdd)
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else {
                panic!("Operator not implemented yet!")
            }
        }
    }
}
