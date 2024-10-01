use orion_numbers::FixedTrait;

pub(crate) fn log2<T, S, +FixedTrait<T, S>, +Copy<T>, +Drop<T>>(
    mut self: Span<T>
) -> Span<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match self.pop_front() {
            Option::Some(ele) => { result_data.append(FixedTrait::log2(*ele)); },
            Option::None(_) => { break; }
        };
    };

    result_data.span()
}

pub(crate) fn exp2<T, S, +FixedTrait<T, S>, +Copy<T>, +Drop<T>>(
    mut self: Span<T>
) -> Span<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match self.pop_front() {
            Option::Some(ele) => { result_data.append(FixedTrait::exp2(*ele)); },
            Option::None(_) => { break; }
        };
    };

    result_data.span()
}

pub(crate) fn sqrt<T, S, +FixedTrait<T, S>, +Copy<T>, +Drop<T>>(
    mut self: Span<T>
) -> Span<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match self.pop_front() {
            Option::Some(ele) => { result_data.append(FixedTrait::sqrt(*ele)); },
            Option::None(_) => { break; }
        };
    };

    result_data.span()
}