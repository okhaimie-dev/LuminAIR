# LuminAIR - Unlocking AI Integrity

LuminAIR is a **Machine Learning** framework that guarantees the integrity of graph-based models with **Zero-Knowledge Circle Stark proofs**.

It enables a prover to cryptographically prove that the AI model's computations have been executed correctly.
Consequently, a verifier can verify these proofs much faster and with fewer resources than by naively re-running the model.

Designed for parallel processing in trace and proof generation, LuminAIR also makes it easy to add support for new backends.

> **⚠️ Disclaimer:** LuminAIR is currently under development 🏗️.

## Example

To see LuminAIR in action, run the provided example:
```bash
$ cd examples/simple
$ cargo run
```

```rust
use std::{fs, path::PathBuf};

use luminair_air::{ops::add::TensorAdd, serde::SerializableTrace, Circuit};
use luminair_compiler::{data::GraphOutputConverter, init_compiler};
use luminal::prelude::*;
use stwo_prover::core::{backend::simd::SimdBackend, vcs::blake2_merkle::Blake2sMerkleChannel};

// =============== Example Overview ===============
// This example demonstrates how to:
// 1. Build a computation graph with tensor operations.
// 2. Compile and execute the graph while generating execution traces.
// 3. Generate ZK proofs from those traces with Stwo prover.
// 4. Verify the proofs to ensure computation integrity.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // =============== Step 1: Building the Graph ===============
    // Create a new computation graph that will hold our tensor operations
    let mut cx = Graph::new();

    // Create three 2x2 tensors initialized with test data
    // In a real application, these could be input features, weights, etc.
    let a = cx.tensor((2, 2)).set(vec![1., 2., 3., 4.]);
    let b = cx.tensor((2, 2)).set(vec![10., 20., 30., 40.]);
    let w = cx.tensor((2, 2)).set(vec![-1., -1., -1., -1.]);

    // Define the computation operations:
    // 1. First add tensors a and b
    let c = a + b;
    // 2. Add tensor w to the result and mark it for retrieval
    let mut d = (c + w).retrieve();

    // Note: No computation happens yet! We've only built a graph of operations.
    // This lazy execution model allows for optimization before running.

    // =============== Step 2: Compilation & Execution ===============
    // Set up a directory to store execution traces
    let trace_registry = PathBuf::from("./traces");

    // Initialize LuminAIR's StwoCompiler with the trace registry
    // This compiler will transform our high-level operations into
    // a format suitable for generating trace of executions.
    let compiler = init_compiler(Some(trace_registry.clone()));

    // Compile the graph - this transforms operations and prepares for execution
    cx.compile(compiler, &mut d);

    // Optional: Visualize the computation graph
    // cx.display();

    // Execute the graph - this is where actual computation happens
    // During execution, traces will be generated and stored for each operation
    cx.execute();

    // Retrieve the final result as f32 values
    let result = cx.get_final_output(d.id);
    println!("result: {:?}", result);

    // =============== Step 3: Proving & Verification ===============
    // For each trace file generated during execution:
    // 1. Load the trace
    // 2. Generate a ZK proof
    // 3. Verify the proof
    //
    // Note: Currently we prove and verify traces independently.
    // Future versions will use proof recursion to combine them.
    for entry in fs::read_dir(trace_registry)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "bin") {
            // Load and convert the trace to SIMD format.
            let loaded = SerializableTrace::load(path.to_str().unwrap())?;
            let trace = loaded.to_trace::<SimdBackend>();

            let config = Default::default();
            println!("==================");

            // Generate a proof for this trace
            println!("Proving trace file: {:?} 🏗️", path);
            let (components, proof) = TensorAdd::prove::<Blake2sMerkleChannel>(&trace, config);
            println!("Proving was successful ✅");

            // Verify the proof to ensure computation integrity
            println!("Verifying proof 🕵️");
            TensorAdd::verify::<Blake2sMerkleChannel>(components, proof, config)
                .unwrap_or_else(|_| panic!("Verification failed for trace {:?}", path));
            println!("Verication was successful 🎉");
        }
    }

    Ok(())
}
```

## Benchmarks

Performance benchmarks for tensor operators [here](https://gizatechxyz.github.io/Luminair/).

## Acknowledgements

A special thanks to the developers and maintainers of the foundational projects that make LuminAIR possible:

- [Luminal](https://github.com/jafioti/luminal): For providing a robust and flexible deep-learning library that serves as the backbone of LuminAIR.
- [Stwo](https://github.com/starkware-libs/stwo): For offering a powerful prover and constraint library.

## License

LuminAIR is released under the [MIT](https://opensource.org/license/mit) License.
