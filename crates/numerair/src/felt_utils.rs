use stwo_prover::core::fields::m31::{M31, P};

/// Integer representation of a value before conversion to field element.
pub type IntegerRep = i32;

// Constants for M31 field arithmetic
pub const P_HALF: u32 = P >> 1;

/// Converts an IntegerRep to a PrimeField element
pub fn integer_rep_to_felt(x: IntegerRep) -> M31 {
    if x >= 0 {
        M31::from_u32_unchecked(x as u32)
    } else {
        -M31::from_u32_unchecked((-x) as u32)
    }
}

/// Converts a PrimeField element to an IntegerRep
pub fn felt_to_integer_rep(x: M31) -> IntegerRep {
    let val = x.0;
    if val > P_HALF {
        -((P - val) as IntegerRep)
    } else {
        val as IntegerRep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_conversion() {
        let x = 42;
        let felt = integer_rep_to_felt(x);
        assert_eq!(felt_to_integer_rep(felt), x);

        let x = -42;
        let felt = integer_rep_to_felt(x);
        assert_eq!(felt_to_integer_rep(felt), x);
    }
}
