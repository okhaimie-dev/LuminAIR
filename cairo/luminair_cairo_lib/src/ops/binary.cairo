use luminair_cairo_lib::Tensor;

pub(crate) fn tensor_add<T, +Add<T>, +Copy<T>, +Drop<T>>(
    mut lhs: Tensor<T>, mut rhs: Tensor<T>
) -> Tensor<T> {
    let mut result_data: Array<T> = ArrayTrait::new();

    loop {
        match lhs.data.pop_front() {
            Option::Some(lhs_ele) => {
                let rhs_ele = rhs.data.pop_front().unwrap();
                result_data.append(*lhs_ele + *rhs_ele);
            },
            Option::None => { break; }
        }
    };

    Tensor { data: result_data.span() }
}
