use std::fmt::Debug;

use luminair_air::{
    components::{Claim, TraceColumn, TraceEval},
    pie::NodeInfo,
};
use luminal::prelude::*;

pub(crate) mod prim;
pub(crate) mod other;

/// Defines an operator that generates execution traces for proof generation.
///
/// Extends the Luminal's `Operator` trait with the ability to produce a trace and claim
/// compatible with the Stwo prover, specific to a given `TraceColumn` type `C`.
pub(crate) trait LuminairOperator<C: TraceColumn + Debug + 'static>: Operator {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> (TraceEval, Claim<C>, Vec<Tensor>);
}

/// Trait to check and invoke trace generation capabilities of an operator.
///
/// Provides methods to determine if an operator supports trace generation and
/// to execute it if available, defaulting to no support.
pub(crate) trait HasProcessTrace<C: TraceColumn + Debug + 'static> {
    /// Returns `true` if the operator supports trace generation, `false` otherwise.
    fn has_process_trace(&self) -> bool {
        false
    }

    /// Calls `process_trace` if supported, returning the result or `None`.
    fn call_process_trace(
        &mut self,
        _inp: Vec<(InputTensor, ShapeTracker)>,
        _node_info: &NodeInfo,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        None
    }
}

/// Wraps a `LuminairOperator` to integrate with the Luminal operator system.
///
/// Bridges operators that generate traces with the standard Luminal's `Operator` trait.
#[derive(Debug)]
struct LuminairWrapper<C: TraceColumn + Debug + 'static>(Box<dyn LuminairOperator<C>>);

impl<C: TraceColumn + Debug + 'static> Operator for LuminairWrapper<C> {
    /// Delegates processing to the wrapped `LuminairOperator`.
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        self.0.process(inp)
    }
}

impl<C: TraceColumn + Debug + 'static> HasProcessTrace<C> for LuminairWrapper<C> {
    /// Indicates that this wrapper supports trace generation.
    fn has_process_trace(&self) -> bool {
        true
    }

    /// Invokes the wrapped operator’s `process_trace` method.
    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        Some(self.0.process_trace(inp, node_info))
    }
}

impl<C: TraceColumn + Debug + 'static> HasProcessTrace<C> for Box<dyn Operator> {
    /// Checks if the boxed operator is a `LuminairWrapper` supporting tracing.
    fn has_process_trace(&self) -> bool {
        if let Some(wrapper) = (**self).as_any().downcast_ref::<LuminairWrapper<C>>() {
            wrapper.has_process_trace()
        } else {
            false
        }
    }

    /// Calls `process_trace` on the boxed operator if it’s a `LuminairWrapper`.
    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
        node_info: &NodeInfo,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        if let Some(wrapper) = (**self).as_any_mut().downcast_mut::<LuminairWrapper<C>>() {
            wrapper.call_process_trace(inp, node_info)
        } else {
            None
        }
    }
}

/// Converts a type into a boxed `Operator` for use in the graph.
///
/// Facilitates wrapping `LuminairOperator` implementations into the Luminal system.
pub(crate) trait IntoOperator<C: TraceColumn + Debug + 'static> {
    fn into_operator(self) -> Box<dyn Operator>;
}

impl<T, C> IntoOperator<C> for T
where
    T: LuminairOperator<C> + 'static,
    C: TraceColumn + Debug + 'static,
{
    /// Wraps the operator in a `LuminairWrapper` and boxes it.
    fn into_operator(self) -> Box<dyn Operator> {
        Box::new(LuminairWrapper(Box::new(self)))
    }
}
