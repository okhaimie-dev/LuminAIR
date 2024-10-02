use std::{
    any::{Any, TypeId},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use rustc_hash::FxHashMap;

use luminal::prelude::*;

use crate::{
    cairo_runner::{CairoRunner, CairoRunnerConfig},
    constants::COMPILED_CAIRO_PATH,
    precomputing::{
        binary::precompile_binary_op,
        helpers::{get_index, get_vec},
    },
    serialization::{serialize_inputs_binary_op, serialize_reduce_op, serialize_unary_op},
    CairoCompilerError,
};

// ====== UNARY ======

macro_rules! cairo_unary_op {
    ($name:ident, $file_name:expr) => {
        #[derive(Clone)]
        pub struct $name {
            sierra_file: PathBuf,
            runner_config: Arc<CairoRunnerConfig>,
        }
        crate::debug_type!($name);

        impl $name {
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

        impl Operator for $name {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                if tensors.len() != 1 {
                    panic!("$name operator requires exactly one input tensor.");
                }

                let inputs = serialize_unary_op(get_vec(&tensors[0].0));

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
    };
}

// ====== BINARY ======
macro_rules! cairo_binary_op {
    ($name:ident, $file_name:expr) => {
        #[derive(Clone)]
        pub struct $name {
            sierra_file: PathBuf,
            runner_config: Arc<CairoRunnerConfig>,
        }
        crate::debug_type!($name);

        impl $name {
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

        impl Operator for $name {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                if tensors.len() != 2 {
                    panic!("$name operator requires exactly two input tensors.");
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
    };
}

// ====== REDUCE ======
macro_rules! cairo_reduce_op {
    ($name:ident, $file_name:expr) => {
        #[derive(Clone)]
        pub struct $name {
            sierra_file: PathBuf,
            runner_config: Arc<CairoRunnerConfig>,
            dim: usize,
        }
        crate::debug_type!($name);

        impl $name {
            pub fn new(sierra_file: PathBuf, runner_config: Arc<CairoRunnerConfig>, dim: usize) -> Self {
                if !sierra_file.exists() {
                    panic!("Sierra file does not exist: {:?}", sierra_file);
                }
                Self {
                    sierra_file,
                    runner_config,
                    dim,
                }
            }
        }

        impl Operator for $name {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                if tensors.len() != 1 {
                    panic!("$name operator requires exactly one input tensor.");
                }

                let sh = tensors[0].1.shape_usize();
                let front_size: usize = sh.iter().take(self.dim).product::<usize>().max(1);
                let back_size = sh.iter().skip(self.dim + 1).product::<usize>().max(1);
                let dim_size: usize = sh[self.dim];

                let inputs = serialize_reduce_op(get_vec(&tensors[0].0), front_size, back_size, dim_size);

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
    };
}

cairo_unary_op!(CairoLog2, "log2.sierra.json");
cairo_unary_op!(CairoExp2, "exp2.sierra.json");
cairo_unary_op!(CairoSin, "sin.sierra.json");
cairo_unary_op!(CairoRecip, "recip.sierra.json");
cairo_unary_op!(CairoSqrt, "sqrt.sierra.json");

cairo_binary_op!(CairoAdd, "add.sierra.json");
cairo_binary_op!(CairoMul, "mul.sierra.json");
cairo_binary_op!(CairoMod, "rem.sierra.json");
cairo_binary_op!(CairoLessThan, "lt.sierra.json");

cairo_reduce_op!(CairoSumReduce, "sum_reduce.sierra.json");
cairo_reduce_op!(CairoMaxReduce, "max_reduce.sierra.json");

#[derive(Clone)]
pub struct CairoConstant {
    pub value: ConstantValue,
    dyn_map: *const FxHashMap<char, usize>,
}

impl core::fmt::Debug for CairoConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CairoConstant({:?})", self.value)
    }
}

impl CairoConstant {
    pub fn new(value: ConstantValue, dyn_map: *const FxHashMap<char, usize>) -> Self {
        Self { value, dyn_map }
    }
}

impl Operator for CairoConstant {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let value = match &self.value {
            ConstantValue::Expression(e) => {
                vec![e.exec(unsafe { self.dyn_map.as_ref().unwrap() }).unwrap() as f32]
            }
            ConstantValue::Float(f) => vec![*f],
        };
        let res = vec![Tensor::new(value)];

        println!("Constant {:?}", res);

        res
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous;

impl Operator for Contiguous {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let inp_data = get_vec(&inp[0].0);
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

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
        _: T,
    ) -> Self::Output {
        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        for id in graph.node_indices().collect::<Vec<_>>() {
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Log2>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("log2.sierra.json");

                *op_ref = Box::new(CairoLog2::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Exp2>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("exp2.sierra.json");

                *op_ref = Box::new(CairoExp2::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Sin>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("sin.sierra.json");

                *op_ref = Box::new(CairoSin::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CairoConstant::new(c.0.clone(), &graph.dyn_map));
            } else if is::<Recip>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("recip.sierra.json");

                *op_ref = Box::new(CairoRecip::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Sqrt>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("sqrt.sierra.json");

                *op_ref = Box::new(CairoSqrt::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Add>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("add.sierra.json");

                *op_ref = Box::new(CairoAdd::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Mul>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("mul.sierra.json");

                *op_ref = Box::new(CairoMul::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Mod>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("rem.sierra.json");

                *op_ref = Box::new(CairoMod::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<LessThan>(op) {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("lt.sierra.json");

                *op_ref = Box::new(CairoLessThan::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(Contiguous)
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("sum_reduce.sierra.json");

                *op_ref = Box::new(CairoSumReduce::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                    *dim,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                let sierra_file = PathBuf::from_str(COMPILED_CAIRO_PATH)
                    .unwrap()
                    .join("max_reduce.sierra.json");

                *op_ref = Box::new(CairoMaxReduce::new(
                    sierra_file,
                    self.runner_config.clone().into(),
                    *dim,
                ));
            }
        }
        Ok(())
    }
}