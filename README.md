# LuminAIR - Unlocking AI Integrity

Luminair is a **Machine Learning** framework that guarantees the integrity of graph-based models using **Zero-Knowledge proofs**. 
It enables a prover to cryptographically prove that the AI model's computations have been executed correctly. 
Consequently, a verifier can verify these proofs much faster and with fewer resources than by naively re-running the model.

Designed for parallel processing in trace and proof generation, Luminair also makes it easy to add support for new zk backends.

> **⚠️ Disclaimer:** Luminair is currently under development and is not recommended for production environments.

## Backends

- **Cairo**: ⚠️ Only for PoC, not optimized for medium/large models.
- **Stwo**: WIP 🏗️

## Benchmarks

Performance benchmarks for tensor operators (stwo-backend support) [here](https://gizatechxyz.github.io/Luminair/).

## Acknowledgements

A special thanks to the developers and maintainers of the foundational projects that make Luminair possible:

- [Luminal](https://github.com/jafioti/luminal): For providing a robust and flexible deep-learning library that serves as the backbone of Luminair.
- [Stwo](https://github.com/starkware-libs/stwo): For offering a powerful prover and constraint library.

## License

Luminair is released under the [MIT](https://opensource.org/license/mit) License.
