# LuminAir - Unlocking AI Integrity

LuminAir is a Deep Learning framework designed to ensure the integrity of Neural Networks using Zero-Knowledge (ZK) proofs. Each node within a Neural Network operates independently on CairoVM, generating execution traces that facilitate parallel ZK proof generation.

Ensuring the integrity of AI is crucial as artificial intelligent systems impact critical sectors like healthcare, finance, autonomous transport or trustless environments like blockchains. ZK proofs enable a prover to guarantee, using cryptographic evidence, that the execution of a model has proceeded correctly. This assurance is provided without the verifier having to re-execute the program.

Built atop [Luminal](https://github.com/jafioti/luminal), a deep-learning library inspired by Tiny-Grad, LuminAir offers simplicity with composable compiler for precomputing, significantly reducing overhead on the Cairo runtime.

> **⚠️ Disclaimer:** LuminAir is currently under development and is not recommended for production environments.

## Table of Contents

- [Getting Started](#getting-started)
- [Design Choices](#design-choices)
  - [Easy to Maintain](#easy-to-maintain)
  - [Proof Parallel Friendly](#proof-parallel-friendly)
  - [Lazy Computation](#lazy-computation)
  - [Why Cairo?](#why-cairo)
- [Roadmap](#roadmap)
- [License](#license)

## Getting Started

No prior knowledge of Cairo is required. You can design your Neural Network directly in Rust and run on CairoVM.

### Example

```rust
use luminair::CairoCompiler;
use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    // Create a new computation graph
    let mut cx = Graph::new();

    // Initialize a linear layer with input size 4 and output size 5
    let model = Linear::new(4, 5, false, &mut cx).initialize();

    // Create an input tensor
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);

    // Forward the tensor through the model
    let mut b = model.forward(a).retrieve();

    // Compile the graph using CairoCompiler
    cx.compile(<CairoCompiler>::default(), &mut b);

    // Execute the computation graph
    cx.execute();

    // Output the results
    println!("B: {:?}", b.data());
}
```

## Design Choices

### Easy to Maintain
By building atop Luminal, LuminAir supports any Neural Network architecture using only 11 primitive operators:

Unary Operators: `Log2, Exp2, Sin, Sqrt, Recip`
Binary Operators: `Add, Mul, Mod, LessThan`
Other Operators: `SumReduce, MaxReduce, Contiguous`

These operators are sufficient to implement complex architectures like transformers and convolutional networks, making the framework straightforward to maintain.

### Proof Parallel Friendly
By decomposing Neural Networks into small operators, LuminAir enables the creation of independent ZK circuits for each graph node. CairoVM generates independent execution traces, allowing node proofs to be generated in parallel.

### Lazy Computation
Expressions like `x + y` are recorded in a directed acyclic computation graph without immediate computation. The actual computation occurs only when `graph.execute()` is invoked. This approach allows precompilation optimizations, such as reducing ZK circuit overhead and fusing operators, handled at the compiler level.

### Why Cairo?

Choosing Cairo is driven by several factors:

- Simplicity: The Cairo programming language is straightforward, facilitating easy development and auditing of operators.
- Advanced ZK Capabilities: [StarkWare](https://starkware.co/) continually advances ZK technologies. [Stwo](https://github.com/starkware-libs/stwo) is a promising prover that unlocking promising features for ZKML, like client-side proving and GPU-based proving.
- Blockchain Verification: Cairo allows proof verification on blockchain platforms, enabling impartial verification by third parties.

## Roadmap
To enhance LuminAir's efficiency and support for large models, the following tasks are planned:

- [] Investigate loading weights directly into bytecode instead of passing them as function parameters.
- [] Implement fusion compilers.
- [] Develop an efficient MatMul compiler.
- [] Execute each node directly from a bootloader to reduce verification costs and enhance model privacy.
- [] Explore parallel trace generation for certain operators at the compiler level.
- [] Implement a zkTree structure to aggregate all proofs of a Neural Network execution.
- [] Ensure that the inputs of a node match the outputs of preceding related nodes to guarantee the integrity of the entire computation process.
- [] Investigate packed arithmetic to perform multiple operations in parallel within the same field.

## Acknowledgements
A special thanks to the developers and maintainers of the foundational projects that make LuminAir possible:

- [Luminal](https://github.com/jafioti/luminal): For providing a robust and flexible deep-learning library that serves as the backbone of LuminAir.
- [CairoVM](https://github.com/lambdaclass/cairo-vm): For offering a powerful VM that enables execution of Cairo programs.


## License
[MIT License](https://opensource.org/license/mit)