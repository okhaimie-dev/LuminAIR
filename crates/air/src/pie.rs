use serde::{Deserialize, Serialize};

use crate::{components::ClaimType, serde::SerializableTrace};

#[derive(Serialize, Deserialize, Debug)]
pub struct LuminairPie {
    pub traces: Vec<Trace>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Trace {
    pub eval: SerializableTrace,
    pub claim: ClaimType,
}
