pub(crate) fn add<T, +Add<T>, +Copy<T>, +Drop<T>>(mut lhs: Span<T>, mut rhs: Span<T>) -> Span<T> {
    let mut result: Array<T> = ArrayTrait::new();

    loop {
        match lhs.pop_front() {
            Option::Some(lhs_ele) => {
                let rhs_ele = rhs.pop_front().unwrap();
                result.append(*lhs_ele + *rhs_ele);
            },
            Option::None => { break; }
        }
    };

    result.span()
}


pub(crate) fn mul<T, +Mul<T>, +Copy<T>, +Drop<T>>(mut lhs: Span<T>, mut rhs: Span<T>) -> Span<T> {
    let mut result: Array<T> = ArrayTrait::new();

    loop {
        match lhs.pop_front() {
            Option::Some(lhs_ele) => {
                let rhs_ele = rhs.pop_front().unwrap();
                result.append(*lhs_ele * *rhs_ele);
            },
            Option::None => { break; }
        }
    };

    result.span()
}
