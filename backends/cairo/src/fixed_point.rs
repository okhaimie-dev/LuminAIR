use lazy_static::lazy_static;
use num_bigint::BigInt;
use num_traits::{Num, ToPrimitive};

const ONE: f32 = 4294967296.0;

lazy_static! {
    pub static ref CAIRO_PRIME_BIGINT: BigInt = BigInt::from_str_radix(
        "800000000000011000000000000000000000000000000000000000000000001",
        16
    )
    .unwrap();
    pub static ref HALF_CAIRO_PRIME_BIGINT: BigInt = &*CAIRO_PRIME_BIGINT >> 1;
}

pub(crate) fn from_float_to_fp(a: f32) -> i64 {
    if a.is_nan() {
        return 0x4e614e; // NaN value in Orion numbers.
    }
    (a * ONE) as i64
}

pub(crate) fn felt_fp_to_float(felt: &BigInt) -> Option<f32> {
    if *felt == BigInt::from(0x4e614e) {
        return Some(f32::NAN);
    }

    felt_to_int(felt).to_f32().map(|n| n / ONE)
}

fn felt_to_int(felt: &BigInt) -> BigInt {
    if felt > &*HALF_CAIRO_PRIME_BIGINT {
        felt - &*CAIRO_PRIME_BIGINT
    } else {
        felt.clone()
    }
}
