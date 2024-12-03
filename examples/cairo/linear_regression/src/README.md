# Linear Regression

This program demonstrates the use of the Luminal and Luminair Cairo crates to create and run a simple linear regression model on CairoVM.

## Running the Program
```shell
cargo run
```

## How It Works
- Graph Initialization: A computation graph is initialized to manage tensors and operations.
- Model Setup: A linear model is created with predefined weights and bias.
- Data Handling: Input data is set dynamically to handle varying batch sizes.
- Compilation and Execution: The graph is compiled using the Luminair's Cairo compiler and then executed on CairoVM.






