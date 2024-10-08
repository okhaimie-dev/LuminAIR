use dfdx::prelude::{Module as DfdxModule, *};
use luminal::{module::Module, prelude::*};
use luminal_nn::{Linear, ReLU};
use rand::{rngs::StdRng, SeedableRng};

use crate::{binary_test, unary_test, CairoCompiler};

luminal::test_imports!();

// =============== UNARY ===============

unary_test!(|a| a.sin(), |a| a.sin(), test_sin, f32);
unary_test!(|a| a.sqrt(), |a| a.sqrt(), test_sqrt, f32);
unary_test!(|a| a.recip(), |a| a.recip(), test_recip, f32);
unary_test!(|a| a * a, |a| a.clone() * a, test_square, f32);
unary_test!(|a| a.ln(), |a| a.ln(), test_ln, f32);
unary_test!(|a| a.log2(), |a| a.ln() / 2_f32.ln(), test_log2, f32);
unary_test!(|a| a.exp2(), |a| (a * 2_f32.ln()).exp(), test_exp2, f32);
unary_test!(
    |a| a.softmax(0),
    |a| a.softmax::<DAxis<0>>(),
    test_softmax,
    f32
);
unary_test!(
    |a| a.mean_norm(0).std_norm(0, 1e-5),
    |a| a.normalize::<DAxis<0>>(1e-5),
    test_norm,
    f32
);

// =============== BINARY ===============

binary_test!(|a, b| a + b, |a, b| a + b, test_add, f32);
binary_test!(|a, b| a - b, |a, b| a - b, test_sub, f32);
binary_test!(|a, b| a * b, |a, b| a * b, test_mul, f32);
binary_test!(|a, b| a / b, |a, b| a / b, test_div, f32);
binary_test!(
    |a, b| a % b,
    |a, b| a.clone() - ((a / b.clone()).to_dtype::<i32>().to_dtype::<f32>() * b),
    test_mod,
    f32
);
binary_test!(|a, b| a.min(b), |a, b| a.minimum(b), test_min, f32);
binary_test!(|a, b| a.max(b), |a, b| a.maximum(b), test_max, f32);

// =============== MOVEMENT ===============

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((3, 4)).set(data.clone());
    let mut b = a.permute((1, 0)).reshape((12, 1)).retrieve();
    let _ = cx.compile(CairoCompiler::default(), &mut b);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<3>, DConst::<4>));
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.as_vec());
}

// =============== REDUCE ===============

#[test]
fn test_sum_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(4 * 512);
    let a = cx.tensor((1, 4, 512));
    a.set(data.clone());
    let mut b = a.sum_reduce(1).retrieve();
    let mut c = a.sum_reduce(0).retrieve();
    let mut d = a.sum_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut c, &mut d));
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<4>, DConst::<512>));
    let d_b = d_a.clone().sum::<_, DAxis<1>>();
    let d_c = d_a.clone().sum::<_, DAxis<0>>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_max_reduce() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((2, 2, 3));
    a.set(data.clone());
    let mut b = a.max_reduce(1).retrieve();
    let mut c = a.max_reduce(0).retrieve();
    let mut d = a.max_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut c, &mut d));
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<2>, DConst::<2>, DConst::<3>));
    let d_b = d_a.clone().max::<_, DAxis<1>>();
    let d_c = d_a.clone().max::<_, DAxis<0>>();
    let d_d = d_a.max::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_mean_reduce() {
    let data = random_vec(1024);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 8, 128)).set(data.clone());
    let mut b = a.mean_reduce(2).retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut b);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<8>, DConst::<128>));
    let d_b = d_a.mean::<_, DAxis<2>>();
    assert_close(&b.data(), &d_b.as_vec());
}

// =============== MATMUL ===============

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a_data = random_vec(3 * 3);
    let b_data = random_vec(3 * 3);
    let a = cx.tensor((3, 3)).set(a_data.clone());
    let b = cx.tensor((3, 3)).set(b_data.clone());
    let mut c = a.matmul(b).retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut c);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<3>, DConst::<3>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<3>, DConst::<3>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set(vec![1., 2., 3., 1., 2., 1., 1., 2., 3., 1., 2., 1.]);
    let b = cx
        .tensor((3, 4))
        .set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);
    let mut c = a.matmul(b).retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut c);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 1.]], [[1., 2., 3.], [1., 2., 1.]]]);
    let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 1., 2.], [-1., -2., 1., 2.]]);
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_matmul_transpose() {
    const M: usize = 16;
    const K: usize = 8;
    const N: usize = 8;
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let a_data = random_vec_rng(M * K, &mut rng);
    let a = cx.tensor((M, K)).set(a_data.clone());
    let b_data = random_vec_rng(K * N, &mut rng);
    let b = cx.tensor((N, K)).set(b_data.clone());
    let a_t_data = random_vec_rng(K * M, &mut rng);
    let a_t = cx.tensor((K, M)).set(a_t_data.clone());
    let b_t_data = random_vec_rng(K * N, &mut rng);
    let b_t = cx.tensor((K, N)).set(b_t_data.clone());

    let mut a_b = a.matmul(b.permute((1, 0))).retrieve();
    let mut a_b_t = a.matmul(b_t).retrieve();
    let mut a_t_b = a_t.permute((1, 0)).matmul(b.permute((1, 0))).retrieve();
    let mut a_t_b_t = a_t.permute((1, 0)).matmul(b_t).retrieve();

    let _ = cx.compile(
        CairoCompiler::default(),
        (&mut a_b, &mut a_b_t, &mut a_t_b, &mut a_t_b_t),
    );
    cx.execute_debug();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<M>, DConst::<K>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<N>, DConst::<K>));
    let d_a_t = d_dev.tensor_from_vec(a_t_data, (DConst::<K>, DConst::<M>));
    let d_b_t = d_dev.tensor_from_vec(b_t_data, (DConst::<K>, DConst::<N>));
    let d_a_b = d_a.clone().matmul(d_b.clone().permute());
    let d_a_b_t = d_a.matmul(d_b_t.clone());
    let d_a_t_b = d_a_t
        .clone()
        .permute::<_, dfdx::shapes::Axes2<1, 0>>()
        .matmul(d_b.permute());
    let d_a_t_b_t = d_a_t
        .permute::<_, dfdx::shapes::Axes2<1, 0>>()
        .matmul(d_b_t);

    assert_close(&a_b.data(), &d_a_b.as_vec());
    assert_close(&a_b_t.data(), &d_a_b_t.as_vec());
    assert_close(&a_t_b.data(), &d_a_t_b.as_vec());
    assert_close(&a_t_b_t.data(), &d_a_t_b_t.as_vec());
}

// =============== NN ===============
#[test]
fn test_relu_and_linear() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let input_data = random_vec(32);
    let w1 = random_vec(32 * 64);
    let w2 = random_vec(32 * 64);
    let batch = cx.named_tensor("Batch", (2, 32)).set(random_vec(32 * 2));
    let a = cx.named_tensor("Single", 32).set(input_data.clone());

    let model = (
        Linear::new(32, 64, false, &mut cx),
        ReLU,
        Linear::new(64, 32, false, &mut cx),
    );
    model.0.weight.set(w1.clone());
    model.2.weight.set(w2.clone());
    let mut b = model.forward(a).retrieve();
    let mut batch_out = model.forward(batch).retrieve();
    cx.execute();

    let unoptimized_b = b.data();
    let unoptimized_batch_out = batch_out.data();
    b.drop();
    batch_out.drop();
    let _ = cx.compile(
        <(GenericCompiler, CairoCompiler)>::default(),
        (&mut b, &mut batch_out),
    );
    cx.execute();

    assert_close_precision(&unoptimized_b, &b.data(), 1e-2);
    assert_close_precision(&unoptimized_batch_out, &batch_out.data(), 1e-2);

    // Test against dfdx
    let dev = Cpu::default();
    let mut model = <(
        dfdx::nn::modules::builders::UnbiasedLinear<32, 64>,
        dfdx::nn::modules::builders::ReLU,
        dfdx::nn::modules::builders::UnbiasedLinear<64, 32>,
    )>::build_on_device(&dev);
    // Set weights
    model.0.weight = dev
        .tensor_from_vec(w1, (DConst::<32>, DConst::<64>))
        .permute();
    model.2.weight = dev
        .tensor_from_vec(w2, (DConst::<64>, DConst::<32>))
        .permute();
    let a = dev.tensor_from_vec(input_data, (DConst::<32>,));
    let out = model.forward(a);

    assert_close_precision(&unoptimized_b, &out.as_vec(), 1e-2);
}

// #[test]
// fn test_transformer_encoder_block() {
//     let mut cx = Graph::new();
//     let model = luminal_nn::TransformerEncoderBlock::new(3, 4, 1, &mut cx);
//     model
//         .attention
//         .w_k
//         .weight
//         .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
//     model
//         .attention
//         .w_q
//         .weight
//         .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
//     model
//         .attention
//         .w_v
//         .weight
//         .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
//     model
//         .attention
//         .w_o
//         .weight
//         .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
//     model
//         .ff
//         .0
//         .weight
//         .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
//     model
//         .ff
//         .2
//         .weight
//         .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

//     let a = cx
//         .tensor(('a', 3))
//         .set_dyn(vec![-1., 2., 3., 3., 3., -1.], (2, 3));
//     let mut b = model.forward(a).retrieve();

//     let _ = cx.compile(<(GenericCompiler, CairoCompiler)>::default(), &mut b);
//     cx.execute();

//     let d_dev = Cpu::default();
//     let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
//         d_dev.build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
//     d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
//     d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
//     d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
//     d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
//     d_model.self_attn.w_o.weight = d_dev
//         .tensor_from_vec(
//             vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
//             (DConst::<3>, DConst::<3>),
//         )
//         .permute();
//     d_model.self_attn.w_k.weight = d_dev
//         .tensor_from_vec(
//             vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
//             (DConst::<3>, DConst::<3>),
//         )
//         .permute();
//     d_model.self_attn.w_q.weight = d_dev
//         .tensor_from_vec(
//             vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
//             (DConst::<3>, DConst::<3>),
//         )
//         .permute();
//     d_model.self_attn.w_v.weight = d_dev
//         .tensor_from_vec(
//             vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
//             (DConst::<3>, DConst::<3>),
//         )
//         .permute();
//     d_model.ff.0 .0.weight = d_dev
//         .tensor_from_vec(
//             vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
//             (DConst::<3>, DConst::<4>),
//         )
//         .permute();
//     d_model.ff.0 .0.bias = d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (DConst::<4>,));
//     d_model.ff.0 .2.weight = d_dev
//         .tensor_from_vec(
//             vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
//             (DConst::<4>, DConst::<3>),
//         )
//         .permute();
//     d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
//     d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
//     d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
//     d_model.norm1.epsilon = 1e-5;
//     d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
//     d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
//     d_model.norm2.epsilon = 1e-5;
//     let d_a = d_dev.tensor_from_vec(vec![-1., 2., 3., 3., 3., -1.], (DConst::<2>, DConst::<3>));
//     let d_b = d_model.forward(d_a);

//     assert_close(&b.data(), &d_b.as_vec());
// }

#[test]
fn test_pool_1d_dims() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor((4, 4)).set(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    // Stride 1
    let mut out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    let _ = cx.compile(<CairoCompiler>::default(), &mut out1);

    cx.execute();

    assert_exact(
        &out1.data(),
        &[
            1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13., 14.,
            15., 14., 15., 16.,
        ],
    );
}

#[test]
fn test_pool_2d() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor((4, 4)).set(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    // 3x3 kernel
    let mut out1 = inp1
        // Pool first dim first by moving it to end
        .permute((1, 0))
        .pool_last_dim(3, 1, 1)
        // Now move other dim to end
        .permute((1, 2, 0))
        .pool_last_dim(3, 1, 1)
        // Now swap middle two dims
        .permute((0, 2, 1, 3))
        // Now merge both pooled dimensions
        .reshape((4, 3, 3))
        .retrieve();

    let _ = cx.compile(<CairoCompiler>::default(), &mut out1);

    cx.execute();

    assert_exact(
        &out1.data(),
        &[
            1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00, 7.00,
            8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00, 14.00, 15.00,
            6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
        ],
    );
}

#[test]
fn test_pool_1d_dilation() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
    // Stride 1
    let mut out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
    // Stride 2
    let mut out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
    // Stride 3
    let mut out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

    let _ = cx.compile(<CairoCompiler>::default(), &mut out1);
    let _ = cx.compile(<CairoCompiler>::default(), &mut out2);
    let _ = cx.compile(<CairoCompiler>::default(), &mut out3);

    cx.execute();

    assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    assert_exact(&out3.data(), &[1., 3.]);
}