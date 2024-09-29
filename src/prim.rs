use std::{
    any::{Any, TypeId},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use luminal::prelude::*;

use crate::{
    cairo_runner::{CairoRunner, CairoRunnerConfig},
    constants::COMPILED_CAIRO_PATH,
    precomputing::binary::precompile_binary_op,
    serialization::serialize_inputs_binary_op,
    CairoCompilerError,
};
use itertools::Itertools;

#[derive(Clone)]
pub struct CairoAdd {
    sierra_file: PathBuf,
    runner_config: Arc<CairoRunnerConfig>,
}
crate::debug_type!(CairoAdd);

impl CairoAdd {
    pub fn new(sierra_file: PathBuf, runner_config: Arc<CairoRunnerConfig>) -> Self {
        if !sierra_file.exists() {
            panic!("Sierra file does not exist: {:?}", sierra_file);
        }
        Self {
            sierra_file,
            runner_config,
        }
    }
}

impl Operator for CairoAdd {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if tensors.len() != 2 {
            panic!("CairoAdd operator requires exactly two input tensors.");
        }

        let (lhs, rhs) = precompile_binary_op(tensors);
        let inputs = serialize_inputs_binary_op(lhs, rhs);

        let cairo_runner = CairoRunner::new((*self.runner_config).clone());
        match cairo_runner.run(self.sierra_file.clone(), inputs, false) {
            Ok(result) => {
                vec![result]
            }
            Err(e) => {
                panic!("Error executing Cairo: {:?}", e);
            }
        }
    }
}

/// Convert all primitive ops to cairo primitive ops.
#[derive(Debug, Default)]
pub struct PrimitiveCompiler {
    runner_config: CairoRunnerConfig,
}

impl PrimitiveCompiler {
    pub fn new(config: CairoRunnerConfig) -> Self {
        Self {
            runner_config: config,
        }
    }
}

impl Compiler for PrimitiveCompiler {
    type Output = Result<(), CairoCompilerError>;

    fn compile<T: luminal::prelude::ToIdsMut>(
        &self,
        graph: &mut luminal::prelude::Graph,
        ids: T,
    ) -> Self::Output {
        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        // Swap primitive ops
        for id in graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Log2>(op) {
                unimplemented!()
            } else if is::<Exp2>(op) {
                unimplemented!()
            } else if is::<Sin>(op) {
                unimplemented!()
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                unimplemented!()
            } else if is::<Recip>(op) {
                unimplemented!()
            } else if is::<Sqrt>(op) {
                unimplemented!()
            } else if is::<Add>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("add.sierra.json");

                *op_ref = Box::new(CairoAdd::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Mul>(op) {
                unimplemented!()
            } else if is::<Mod>(op) {
                unimplemented!()
            } else if is::<LessThan>(op) {
                unimplemented!()
            } else if is::<Contiguous>(op) {
                unimplemented!()
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                unimplemented!()
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                unimplemented!()
            }
        }
        Ok(())
    }
}
