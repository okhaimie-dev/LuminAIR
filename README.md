# Luminair - Unlocking AI Integrity

Luminair is a **Machine Learning** framework that guarantees the integrity of graph-based models using **Zero-Knowledge proofs**. 
It enables a prover to cryptographically prove that the AI model's computations have been executed correctly. 
Consequently, a verifier can verify these proofs much faster and with fewer resources than by naively re-running the model.

Designed for parallel processing in trace and proof generation, Luminair also makes it easy to add support for new zkVMs.

> **⚠️ Disclaimer:** Luminair is currently under development and is not recommended for production environments.

## Table of Contents

- [Add new zkVMs](#add-new-zkvms)
- [Features](#features)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Add new zkVMs

Luminair has been designed to support a wide range of zkVMs. Adding new VMs is easy, just implement 11 primitive operators to support any model.

### Primitive operators

- **Unary Operators:** `Log2, Exp2, Sin, Sqrt, Recip`
- **Binary Operators:** `Add, Mul, Mod, LessThan`
- **Other Operators:** `SumReduce, MaxReduce, Contiguous`

These ops are enough to support transformers, convnets, etc.

### Supported zkVMs

| zkVMs | Status |
| ----- | ------ |
| Cairo | ✅     |
| SP1   | 👀     |
| Risc0 | 👀     |
| Noir  | 👀     |

## Features

- **Proof Parallelism:** Each node in the computation graph can be proved independently, enabling parallel proof generation.
- **Extensible zkVM Support:** Easily add support for new zkVMs by implementing only 11 primitive operators to support any model.
- **Precompilation Optimizations:** Reduce ZK circuit overhead and fuse operators at the compiler level for enhanced performance.
- **Mainstream-Language Support:** Design ML models in Rust (with Python support coming soon) to onboard a wide range of developers.

## Usage

Below is a simple example demonstrating how to create and execute on CairoVM a computation graph using Luminair in Rust:

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
    cx.compile(CairoCompiler::default(), &mut b);

    // Execute the computation graph
    cx.execute();

    // Output the results
    println!("B: {:?}", b.data());
}
```

## Key Concepts

- **Graph-Based Execution:** Expressions like `x + y` are recorded in a directed acyclic computation graph without immediate computation. The actual computation occurs only when `graph.execute()` is invoked. This allows lazy computation to reduce the overhead on runtime.
- **Parallel Proof Generation:** By decomposing the model into small operators, Luminair enables the creation of independent and simple ZK circuits for each graph node. A zkVM then generates independent execution traces, allowing node proofs to be generated in parallel.

## Roadmap

To enhance Luminair's efficiency and support for large models, the following features are planned:

- **Fusion Compilers:** Implement fusion compilers for certain operators to optimize performance.
- **Efficient MatMul Compiler:** Develop optimized compilers for matrix multiplication operations.
- **Bootloader Integration:** Execute each circuit from a [bootloader](https://github.com/starkware-libs/cairo-lang/tree/master/src/starkware/cairo/bootloaders) to reduce verification costs and enhance model privacy.
- **Trace Generation Parallelization:** Enhance parallelization in trace generation for faster model execution.
- **Dataflow Integrity:** Ensure that the inputs of a node match the outputs of preceding related nodes to guarantee dataflow integrity.
- **zkTree Structure:** Implement a zkTree structure to aggregate all proofs efficiently.
- **Packed Arithmetic:** Investigate packed arithmetic to perform multiple operations in parallel within the same finite field.
- **ONNX Support:** Enable support for ONNX models to broaden compatibility.
- **Python SDK:** Create a Python SDK to allow model design in Python, attracting a larger developer base.

## Acknowledgements

A special thanks to the developers and maintainers of the foundational projects that make Luminair possible:

- [Luminal](https://github.com/jafioti/luminal): For providing a robust and flexible deep-learning library that serves as the backbone of Luminair.
- [CairoVM](https://github.com/lambdaclass/cairo-vm): For offering a powerful VM that enables execution of Cairo programs.

## License

Luminair is released under the [MIT](https://opensource.org/license/mit) License.

_Feel free to contribute to Luminair by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated!_
