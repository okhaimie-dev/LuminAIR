# LuminAIR - Unlocking AI Integrity

LuminAIR is a Machine Learning framework that leverages [Circle Stark Proofs](https://eprint.iacr.org/2024/278) to ensure the integrity of computational graphs. It allows a prover to cryptographically demonstrate that a computational graph has been executed correctly. A verifier can then validate this proof using fewer resources than re-executing the graph.

> **⚠️ Disclaimer:** LuminAIR is currently under development 🏗️.

## RISC-style architecture

In its initial phase, LuminAIR supports a primitive set of 11 operators, sufficient to cover a large type of models (linear regression, convolutional networks, transformers, ...).

| Operator   | Status |
| ---------- | ------ |
| Log2       | ⏳     |
| Exp2       | ⏳     |
| Sin        | ⏳     |
| Sqrt       | ⏳     |
| Recip      | ⏳     |
| Add        | ✅     |
| Mul        | ⏳     |
| Mod        | ⏳     |
| LessThan   | ⏳     |
| SumReduce  | ⏳     |
| MaxReduce  | ⏳     |
| Contiguous | ✅     |

Future phases will focus on adding fused and specialized operators for improved efficiency.

## Example

To see LuminAIR in action, run the provided example:

```bash
$ cd examples/simple
$ cargo run
```

```rust
use luminair_compiler::{graph::LuminairGraph, StwoCompiler};
use luminal::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cx = Graph::new();

    // Define tensors
    let a = cx.tensor((2, 2)).set(vec![1.0, 2.0, 3.0, 4.0]);
    let b = cx.tensor((2, 2)).set(vec![10.0, 20.0, 30.0, 40.0]);
    let w = cx.tensor((2, 2)).set(vec![-1.0, -1.0, -1.0, -1.0]);

    // Build computation graph
    let c = a + b;
    let mut d = (c + w).retrieve();

    // Compile the computation graph
    cx.compile(<(GenericCompiler, StwoCompiler)>::default(), &mut d);

    // Execute and generate a trace of the computation graph
    let trace = cx.gen_trace();

    // Generate proof and verify
    let proof = cx.prove(trace)?;
    cx.verify(proof)?;

    Ok(())
}
```

## Acknowledgements

A special thanks to the developers and maintainers of the foundational projects that make LuminAIR possible:

- [Luminal](https://github.com/jafioti/luminal): For providing a robust and flexible deep-learning library that serves as the backbone of LuminAIR.
- [Stwo](https://github.com/starkware-libs/stwo): For offering a powerful prover and constraint library.
- [Brainfuck-Stwo](https://github.com/kkrt-labs/stwo-brainfuck): Inspiration for creating AIR with the Stwo library.

## License

LuminAIR is is open-source software released under the [MIT](https://opensource.org/license/mit) License.
