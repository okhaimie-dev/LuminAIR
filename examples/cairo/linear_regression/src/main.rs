use luminair_cairo::CairoCompiler;
use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    let mut cx = Graph::new();

    const INPUT_FEATURES: usize = 4;
    const OUTPUT_DIM: usize = 1;

    let mut input = cx.named_tensor("Input", (1, INPUT_FEATURES));

    let model = Linear::new_permuted(INPUT_FEATURES, OUTPUT_DIM, true, &mut cx);

    model.weight.set(vec![0.5, -0.3, 0.8, 0.1]);
    model.bias.as_ref().unwrap().set(vec![0.2]);

    let mut model_weights = params(&model);
    cx.keep_tensors(&model_weights);

    let mut output = model.forward(input).retrieve();

    println!("Compiling graph...");
    let _ = cx.compile(
        (GenericCompiler::default(), CairoCompiler::default()),
        (&mut input, &mut output, &mut model_weights),
    );

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];

    input.set_dyn(input_data, (2, INPUT_FEATURES));

    println!("Running inference...");
    cx.execute_debug();

    let predictions = output.data();
    println!("\nPredictions:");
    for (i, pred) in predictions.iter().enumerate() {
        println!("Sample {}: {:.3}", i, pred);
    }
}
