use std::fmt::Debug;

use luminair_air::components::{Claim, TraceColumn, TraceEval};
use luminal::prelude::*;

pub(crate) mod prim;

pub(crate) trait LuminairOperator<C: TraceColumn + Debug + 'static>: Operator {
    fn process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> (TraceEval, Claim<C>, Vec<Tensor>);
}

pub(crate) trait HasProcessTrace<C: TraceColumn + Debug + 'static> {
    fn has_process_trace(&self) -> bool {
        false
    }

    fn call_process_trace(
        &mut self,
        _inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        None
    }
}

#[derive(Debug)]
struct LuminairWrapper<C: TraceColumn + Debug + 'static>(Box<dyn LuminairOperator<C>>);

impl<C: TraceColumn + Debug + 'static> Operator for LuminairWrapper<C> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        println!("Wrapper process called");
        self.0.process(inp)
    }
}

impl<C: TraceColumn + Debug + 'static> HasProcessTrace<C> for LuminairWrapper<C> {
    fn has_process_trace(&self) -> bool {
        true
    }

    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        Some(self.0.process_trace(inp))
    }
}

impl<C: TraceColumn + Debug + 'static> HasProcessTrace<C> for Box<dyn Operator> {
    fn has_process_trace(&self) -> bool {
        if let Some(wrapper) = (**self).as_any().downcast_ref::<LuminairWrapper<C>>() {
            wrapper.has_process_trace()
        } else {
            false
        }
    }

    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(TraceEval, Claim<C>, Vec<Tensor>)> {
        if let Some(wrapper) = (**self).as_any_mut().downcast_mut::<LuminairWrapper<C>>() {
            wrapper.call_process_trace(inp)
        } else {
            None
        }
    }
}

pub(crate) trait IntoOperator<C: TraceColumn + Debug + 'static> {
    fn into_operator(self) -> Box<dyn Operator>;
}

impl<T, C> IntoOperator<C> for T
where
    T: LuminairOperator<C> + 'static,
    C: TraceColumn + Debug + 'static,
{
    fn into_operator(self) -> Box<dyn Operator> {
        println!("Converting LuminairOperator to Operator");
        Box::new(LuminairWrapper(Box::new(self)))
    }
}
