use luminair_air::gen::Trace;
use luminal::prelude::*;

pub mod prim;

pub trait LuminairOperator: Operator {
    fn process_trace(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> (Vec<Tensor>, Trace);
}

pub trait HasProcessTrace {
    fn has_process_trace(&self) -> bool {
        false
    }
    fn call_process_trace(
        &mut self,
        _inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(Vec<Tensor>, Trace)> {
        None
    }
}

#[derive(Debug)]
pub struct LuminairWrapper(pub Box<dyn LuminairOperator>);

impl Operator for LuminairWrapper {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        println!("Wrapper process called");
        self.0.process(inp)
    }
}

impl HasProcessTrace for LuminairWrapper {
    fn has_process_trace(&self) -> bool {
        true
    }

    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(Vec<Tensor>, Trace)> {
        Some(self.0.process_trace(inp))
    }
}

impl HasProcessTrace for Box<dyn Operator> {
    fn has_process_trace(&self) -> bool {
        if let Some(wrapper) = (**self).as_any().downcast_ref::<LuminairWrapper>() {
            wrapper.has_process_trace()
        } else {
            false
        }
    }

    fn call_process_trace(
        &mut self,
        inp: Vec<(InputTensor, ShapeTracker)>,
    ) -> Option<(Vec<Tensor>, Trace)> {
        if let Some(wrapper) = (**self).as_any_mut().downcast_mut::<LuminairWrapper>() {
            wrapper.call_process_trace(inp)
        } else {
            None
        }
    }
}

pub trait IntoOperator {
    fn into_operator(self) -> Box<dyn Operator>;
}

impl<T: LuminairOperator + 'static> IntoOperator for T {
    fn into_operator(self) -> Box<dyn Operator> {
        println!("Converting LuminairOperator to Operator");
        Box::new(LuminairWrapper(Box::new(self)))
    }
}
