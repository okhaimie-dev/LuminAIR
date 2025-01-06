pub mod data;
pub mod prim;
pub mod fixed_point;

pub type StwoCompiler<'a> = (prim::PrimitiveCompiler,);

#[cfg(test)]
mod tests {
    use luminal::prelude::*;

    use super::StwoCompiler;

    #[test]
    fn test_stwo_add() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 2)).set(vec![1., 2., 3., 4.]);
        let b = cx.tensor((2, 2)).set(vec![10., 20., 30., 40.]);

        let w = cx.tensor((2, 2)).set(vec![1., 1., 1., 1.]);

        let c = a + b;
        let mut d = (c + w).retrieve();

        cx.compile(<(GenericCompiler, StwoCompiler)>::default(), &mut d);
        cx.execute();
    }
}
