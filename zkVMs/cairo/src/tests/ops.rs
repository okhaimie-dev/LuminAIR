use dfdx::prelude::{Module as DfdxModule, *};
use luminal::{module::Module, prelude::*};
use luminal_nn::{Conv1D, LayerNorm, Linear, ReLU};
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

#[test]
fn test_rotate() {
    let mut cx = Graph::new();
    const B: usize = 2;
    const F: usize = 3;
    const D: usize = 4;
    let data = random_vec(D * B * F);
    let a = cx.tensor((F, B, D)).set(data.clone()).permute((1, 0, 2));
    let x1 = a.slice((.., .., ..D / 2));
    let x2 = a.slice((.., .., D / 2..));
    let mut rotated_a = (-x2).concat_along(x1, 1).retrieve();
    cx.execute();
    let unopt = rotated_a.data();
    rotated_a.drop();

    let _ = cx.compile(CairoCompiler::default(), &mut rotated_a);
    cx.execute_debug();
    assert_close(&unopt, &rotated_a.data());
}

#[test]
fn test_constant() {
    let mut cx = Graph::new();
    let a = cx.constant('a');
    let mut a = (a * a).retrieve();
    let _ = cx.compile(CairoCompiler::default(), &mut a);

    cx.set_dyn_dim('a', 10);
    cx.execute();
    assert_exact(&a.data(), &[100.0]);
    a.drop();
    cx.set_dyn_dim('a', 25);
    cx.execute();
    assert_exact(&a.data(), &[625.0]);
}

#[test]
fn test_slice() {
    let data = random_vec(256);
    let mut cx = Graph::new();
    let mut a = cx.tensor(256).set(data.clone());
    let c = a.slice(..20).contiguous().retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut a);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<256>,))
        .to_dtype::<f32>();
    let d_c = d_a.slice((..20,)).to_dtype::<f32>();

    assert_exact(&c.data(), &d_c.as_vec());
}

#[test]
fn test_pad() {
    // Pad a 8x2 mat to 10x4
    let data = random_vec(8 * 2);
    let mut cx = Graph::new();
    let a = cx.tensor((8, 2)).set(data.clone());
    let mut c = a.pad(((0, 2), (0, 2))).contiguous().retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (8, 2)).to_dtype::<f32>();
    // There is no pad function in dfdx, so we concat with zero tensors
    let d_b = (d_a, d_dev.zeros_like(&(2, 2))).concat_along(DAxis::<0>);
    let d_c = (d_b, d_dev.zeros_like(&(10, 2))).concat_along(DAxis::<1>);

    assert_exact(&c.data(), &d_c.to_dtype::<f32>().as_vec());
}

#[test]
fn test_pad_contig() {
    let m = 13;
    let k = 24;
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let a_data = random_vec_rng(m * k, &mut rng);
    let mut a = cx.tensor(('M', 'K')).set_dyn(a_data, (m, k)).retrieve();
    let mut b = a
        .pad(((0, 0), (0, Expression::from(24) - 'K')))
        .contiguous()
        .retrieve();
    let mut c = (a.slice((.., ..k)) / 1.0).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut a, &mut b, &mut c));
    cx.execute();

    // Close because b and c are going through 16 bits, while a is not
    assert_close(&a.data(), &b.data());
    assert_close(&a.data(), &c.data());
}

#[test]
fn test_movement() {
    let data = random_vec(32);
    let mut cx = Graph::new();
    let a = cx.tensor(32).set(data.clone());
    let b = a.pad((0, 10)).contiguous().retrieve();
    let mut c = b.slice((..25,)).contiguous().retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev
        .tensor_from_vec(data, (DConst::<32>,))
        .to_dtype::<f32>();
    let d_c = d_a.slice((..25,)).to_dtype::<f32>();

    assert_exact(&c.data(), &d_c.as_vec());
}

#[test]
fn test_slice_add() {
    let mut cx = Graph::new();
    let a = cx.tensor(256).set(random_array::<256>());
    let mut b = (a.slice(0..64) + a.slice(64..128) + a.slice(128..192) + a.slice(192..256))
        .expand(0, 4)
        .retrieve();

    let _ = cx.compile(CairoCompiler::default(), &mut b);
    cx.execute();
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
fn test_matmul_simple() {
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
    let mut cx = Graph::new();
    let input_data = random_vec(8);
    let w1 = random_vec(8 * 16);
    let w2 = random_vec(16 * 8);
    let batch = cx.named_tensor("Batch", (2, 8)).set(random_vec(8 * 2));
    let a = cx.named_tensor("Single", 8).set(input_data.clone());

    let model = (
        Linear::new(8, 16, false, &mut cx),
        ReLU,
        Linear::new(16, 8, false, &mut cx),
    );
    model.0.weight.set(w1.clone());
    model.2.weight.set(w2.clone());
    let mut b = model.forward(a).retrieve();
    let mut batch_out = model.forward(batch).retrieve();
    cx.execute_debug();

    let unoptimized_b = b.data();
    let unoptimized_batch_out = batch_out.data();
    b.drop();
    batch_out.drop();
    let _ = cx.compile(
        <(GenericCompiler, CairoCompiler)>::default(),
        (&mut b, &mut batch_out),
    );
    cx.execute_debug();

    assert_close_precision(&unoptimized_b, &b.data(), 1e-2);
    assert_close_precision(&unoptimized_batch_out, &batch_out.data(), 1e-2);

    // Test against dfdx
    let dev = Cpu::default();
    let mut model = <(
        dfdx::nn::modules::builders::UnbiasedLinear<8, 16>,
        dfdx::nn::modules::builders::ReLU,
        dfdx::nn::modules::builders::UnbiasedLinear<16, 8>,
    )>::build_on_device(&dev);
    // Set weights
    model.0.weight = dev
        .tensor_from_vec(w1, (DConst::<8>, DConst::<16>))
        .permute();
    model.2.weight = dev
        .tensor_from_vec(w2, (DConst::<16>, DConst::<8>))
        .permute();
    let a = dev.tensor_from_vec(input_data, (DConst::<8>,));
    let out = model.forward(a);

    assert_close_precision(&unoptimized_b, &out.as_vec(), 1e-2);
}

#[test]
fn test_transformer_encoder_block() {
    let mut cx = Graph::new();
    let model = luminal_nn::TransformerEncoderBlock::new(3, 4, 1, &mut cx);

    model
        .attention
        .w_k
        .weight
        .set(vec![0.1, 0.22, 0.03, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]);
    model
        .attention
        .w_q
        .weight
        .set(vec![0.3, 0.2, 0.3, 0.13, 0.2, 0.3, 0.3, 0.2, 0.3]);
    model
        .attention
        .w_v
        .weight
        .set(vec![-0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3]);
    model
        .attention
        .w_o
        .weight
        .set(vec![0.1, 0.22, 0.03, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]);
    model.ff.0.weight.set(vec![
        -0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3, 0.11, 0.2, 0.3,
    ]);
    model.ff.2.weight.set(vec![
        -0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3, 0.3, -0.1, 0.2,
    ]);

    let a = cx
        .tensor(('a', 3))
        .set_dyn(vec![-0.1, 0.2, 0.3, 0.3, 0.3, -0.1], (2, 3));

    let mut b = model.forward(a).retrieve();

    let _ = cx.compile(<(GenericCompiler, CairoCompiler)>::default(), &mut b);
    cx.execute_debug();

    let d_dev = Cpu::default();
    let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
        d_dev.build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();

    d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_o.bias.copy_from(&[0.0, 0.0, 0.0]);

    d_model.self_attn.w_o.weight = d_dev
        .tensor_from_vec(
            vec![0.1, 0.22, 0.03, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_k.weight = d_dev
        .tensor_from_vec(
            vec![0.1, 0.22, 0.03, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_q.weight = d_dev
        .tensor_from_vec(
            vec![0.3, 0.2, 0.3, 0.13, 0.2, 0.3, 0.3, 0.2, 0.3],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_v.weight = d_dev
        .tensor_from_vec(
            vec![-0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.ff.0 .0.weight = d_dev
        .tensor_from_vec(
            vec![
                -0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3, 0.11, 0.2, 0.3,
            ],
            (DConst::<3>, DConst::<4>),
        )
        .permute();
    d_model.ff.0 .0.bias = d_dev.tensor_from_vec(vec![0.0, 0.0, 0.0, 0.0], (DConst::<4>,));
    d_model.ff.0 .2.weight = d_dev
        .tensor_from_vec(
            vec![
                -0.1, 0.12, 0.03, -0.1, 0.2, -0.3, 0.11, 0.2, 0.3, 0.3, -0.1, 0.2,
            ],
            (DConst::<4>, DConst::<3>),
        )
        .permute();
    d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0.0, 0.0, 0.0], (DConst::<3>,));
    d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1.0, 1.0, 1.0], (DConst::<3>,));
    d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1.0, 1.0, 1.0], (DConst::<3>,));
    d_model.norm1.epsilon = 1e-5;
    d_model.norm2.beta = d_dev.tensor_from_vec(vec![0.0, 0.0, 0.0], (DConst::<3>,));
    d_model.norm1.beta = d_dev.tensor_from_vec(vec![0.0, 0.0, 0.0], (DConst::<3>,));
    d_model.norm2.epsilon = 1e-5;

    let d_a = d_dev.tensor_from_vec(
        vec![-0.1, 0.2, 0.3, 0.3, 0.3, -0.1],
        (DConst::<2>, DConst::<3>),
    );
    let d_b = d_model.forward(d_a);

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_pool_1d_dims() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor((4, 4)).set(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    // Stride 1
    let mut out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    let _ = cx.compile(<CairoCompiler>::default(), &mut out1);

    cx.execute_debug();

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

    cx.execute_debug();

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

    cx.execute_debug();

    assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    assert_exact(&out3.data(), &[1., 3.]);
}

#[test]
fn test_conv2d() {
    let mut cx = Graph::new();

    const CH_IN: usize = 5;
    const CH_OUT: usize = 2;
    const KERNELX: usize = 2;
    const KERNELY: usize = 2;
    const STRIDEX: usize = KERNELX;
    const STRIDEY: usize = KERNELY;
    const DILATIONX: usize = 1;
    const DILATIONY: usize = 1;
    const DIMX_IN: usize = 16;
    const DIMY_IN: usize = 9;

    let inp1 = cx.tensor((CH_IN, DIMX_IN, DIMY_IN)).set(vec![
        8., 8., 5., 7., 0., 6., 5., 3., 0., 7., 0., 6., 6., 7., 7., 5., 0., 6., 9., 4., 0., 8., 8.,
        5., 7., 6., 2., 8., 9., 5., 0., 3., 1., 1., 8., 4., 1., 1., 5., 6., 9., 3., 2., 9., 4., 7.,
        1., 0., 7., 7., 4., 9., 5., 0., 4., 7., 4., 7., 8., 8., 4., 8., 4., 7., 9., 3., 7., 9., 5.,
        8., 5., 9., 0., 9., 5., 6., 8., 9., 5., 4., 1., 9., 7., 2., 2., 7., 9., 3., 1., 2., 8., 4.,
        0., 8., 0., 5., 6., 7., 7., 4., 3., 4., 6., 8., 3., 7., 8., 8., 7., 1., 5., 1., 8., 0., 1.,
        1., 7., 3., 2., 1., 0., 4., 5., 4., 3., 2., 5., 4., 2., 4., 1., 9., 4., 1., 9., 7., 7., 1.,
        2., 6., 3., 4., 1., 1., 6., 6., 8., 2., 7., 7., 9., 0., 9., 0., 1., 4., 2., 4., 9., 6., 8.,
        6., 1., 6., 3., 8., 3., 4., 5., 0., 2., 1., 8., 2., 2., 8., 7., 0., 7., 7., 3., 4., 5., 0.,
        7., 2., 1., 1., 4., 2., 9., 9., 6., 1., 5., 4., 6., 9., 5., 4., 1., 9., 1., 5., 5., 5., 8.,
        8., 0., 1., 3., 0., 8., 8., 5., 1., 6., 1., 5., 6., 4., 4., 4., 0., 1., 1., 5., 1., 7., 2.,
        3., 5., 5., 4., 9., 1., 3., 7., 6., 7., 1., 5., 3., 8., 6., 6., 6., 7., 3., 2., 2., 8., 1.,
        3., 0., 2., 7., 6., 5., 7., 5., 7., 8., 1., 2., 2., 5., 0., 2., 9., 1., 5., 3., 8., 7., 9.,
        7., 2., 8., 8., 8., 6., 3., 2., 7., 7., 0., 3., 7., 8., 3., 7., 2., 3., 2., 7., 5., 5., 6.,
        0., 9., 0., 9., 9., 1., 8., 7., 9., 6., 8., 7., 5., 4., 9., 5., 6., 3., 2., 8., 3., 0., 6.,
        3., 8., 3., 1., 8., 7., 2., 0., 7., 7., 7., 7., 8., 0., 4., 9., 8., 2., 0., 4., 4., 3., 5.,
        5., 3., 0., 3., 6., 3., 1., 2., 9., 9., 6., 8., 1., 2., 6., 8., 6., 0., 0., 2., 8., 8., 5.,
        0., 5., 9., 0., 8., 1., 1., 3., 5., 9., 3., 5., 8., 6., 3., 2., 9., 4., 8., 3., 9., 5., 2.,
        9., 0., 1., 6., 8., 0., 3., 0., 1., 2., 1., 0., 1., 4., 1., 1., 0., 6., 9., 2., 7., 2., 6.,
        0., 4., 8., 2., 6., 7., 2., 2., 7., 4., 5., 8., 1., 4., 7., 5., 9., 7., 2., 5., 9., 1., 6.,
        1., 7., 9., 5., 6., 9., 3., 5., 1., 6., 1., 3., 3., 9., 3., 9., 0., 1., 8., 1., 9., 8., 5.,
        3., 4., 4., 1., 5., 5., 4., 4., 5., 8., 7., 1., 1., 7., 3., 9., 0., 1., 3., 4., 8., 4., 0.,
        5., 6., 2., 0., 7., 8., 2., 6., 2., 9., 6., 2., 0., 3., 7., 5., 7., 1., 8., 5., 5., 9., 1.,
        0., 3., 5., 7., 5., 3., 2., 8., 6., 3., 0., 5., 8., 5., 7., 8., 8., 2., 9., 0., 1., 8., 6.,
        0., 3., 2., 5., 2., 9., 8., 9., 6., 2., 0., 3., 2., 5., 9., 1., 3., 6., 5., 2., 8., 2., 2.,
        1., 8., 6., 4., 1., 6., 0., 7., 3., 0., 9., 6., 5., 5., 5., 2., 4., 2., 8., 3., 0., 6., 3.,
        8., 8., 4., 9., 4., 7., 0., 3., 5., 1., 4., 6., 0., 0., 5., 9., 7., 8., 6., 7., 0., 6., 7.,
        0., 5., 8., 8., 6., 4., 6., 0., 2., 3., 2., 8., 7., 5., 9., 6., 6., 2., 0., 4., 4., 4., 4.,
        2., 7., 5., 3., 2., 6., 3., 7., 0., 7., 2., 5., 1., 4., 4., 5., 1., 6., 7., 5., 7., 0., 7.,
        8., 4., 7., 3., 9., 1., 7., 5., 6., 1., 0., 2., 0., 0., 5., 5., 8., 8., 7., 3., 7., 2., 9.,
        3., 8., 4., 5., 3., 8., 5., 2., 0., 2., 0., 5., 9., 0., 3., 8., 0., 4., 1., 8., 4., 8., 9.,
        1., 1., 4., 5., 0., 2., 0., 9., 4., 2., 3., 9., 0., 7., 3., 1., 5., 9., 1., 6., 5., 4., 2.,
        1., 2., 1., 1., 4., 7., 2.,
    ]);

    let model = luminal_nn::Conv2D::new(
        CH_IN,
        CH_OUT,
        (KERNELX, KERNELY),
        (STRIDEX, STRIDEY),
        (DILATIONX, DILATIONY),
        false,
        &mut cx,
    );
    model.weight.set(vec![
        0.1600, 0.2000, 0.1900, -0.1100, 0.0100, -0.0300, -0.1200, -0.0800, -0.1300, -0.0300,
        0.1600, -0.1700, -0.0000, 0.1900, 0.1300, 0.0300, -0.1500, 0.0900, 0.0100, 0.0200, 0.1500,
        0.0700, -0.0800, 0.1700, 0.1000, -0.0700, 0.1600, -0.1600, -0.1900, -0.0500, -0.2100,
        0.0100, -0.2000, 0.2100, -0.0400, -0.1400, 0.1500, 0.0500, -0.1700, 0.1400,
    ]);

    let mut out1 = model.forward(inp1).retrieve();

    let _ = cx.compile(<(GenericCompiler, CairoCompiler)>::default(), &mut out1);
    cx.execute_debug();

    assert_close(
        &out1.data(),
        &[
            3.9600, -0.3300, -1.7800, 4.0400, 1.5300, 0.2900, 2.8700, 3.0000, 0.9600, -1.8700,
            4.5900, 3.9700, 1.2800, 1.1800, 3.7800, 2.8500, 0.5500, 0.5600, 3.9800, 1.3200,
            -0.7100, -0.6500, 4.3900, 0.4000, 1.0300, 0.9800, 3.1200, 2.7400, 2.5100, 0.1200,
            1.8500, 2.0000, -0.7900, 1.0700, -0.3900, -0.8100, -2.5100, -2.9700, 0.2100, 1.8400,
            -0.7700, -0.3900, 1.2200, 0.1900, 4.1700, -4.3600, -1.8600, 0.4800, -2.4400, 2.6300,
            1.5000, -1.9700, 1.2800, -2.8200, -2.3200, 0.2200, -0.3800, 2.1800, -0.8200, -1.5700,
            1.2000, -3.4200, -1.6700, 0.9000,
        ],
    );
}

#[test]
fn test_conv1d_pad_stride() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    const CH_IN: usize = 8;
    const CH_OUT: usize = 16;
    const KERNEL: usize = 3;
    const STRIDE: usize = 1;
    const PADDING: usize = 1;
    const DIM_IN: usize = 6;
    let kernel_data = random_vec_rng(KERNEL * CH_IN * CH_OUT, &mut rng);
    let input_data = random_vec_rng(CH_IN * DIM_IN, &mut rng);

    let model = Conv1D::new(CH_IN, CH_OUT, KERNEL, STRIDE, 1, PADDING, false, &mut cx);
    model.weight.set(kernel_data.clone());

    let inp1 = cx
        .tensor((1, CH_IN, 's'))
        .set_dyn(input_data.clone(), (1, CH_IN, DIM_IN));

    let mut out1 = model.forward(inp1).retrieve();
    let _ = cx.compile(CairoCompiler::default(), &mut out1);
    cx.execute_debug();

    let input =
        candle_core::Tensor::from_vec(input_data, (1, CH_IN, DIM_IN), &candle_core::Device::Cpu)
            .unwrap();
    let kernel = candle_core::Tensor::from_vec(
        kernel_data,
        (CH_OUT, CH_IN, KERNEL),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let output = input.conv1d(&kernel, PADDING, STRIDE, 1, 1).unwrap();

    assert_close(
        &out1.data(),
        &output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}

#[test]
fn test_embedding() {
    let mut cx = Graph::new();
    let batch = cx
        .named_tensor("Batch", (2, 3))
        .set(vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0])
        .keep();
    let a = cx.named_tensor("Single", 3).set(vec![1.0, 0.0, 1.0]).keep();

    let model = luminal_nn::Embedding::new(3, 4, &mut cx);
    model
        .weight
        .set(vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.]);
    let mut b = model.forward(a).retrieve();
    let mut batch_out = model.forward(batch).retrieve();

    let _ = cx.compile(CairoCompiler::default(), (&mut b, &mut batch_out));
    cx.execute();

    let d_dev = Cpu::default();
    let mut d_model: modules::Embedding<3, 4, f32, Cpu> =
        <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
    d_model.weight = d_dev.tensor_from_vec(
        vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.],
        (DConst::<3>, DConst::<4>),
    );
    let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (DConst::<3>,));
    let d_batch = d_dev.tensor_from_vec(vec![1, 0, 2, 1, 0, 1], (DConst::<2>, DConst::<3>));

    let d_b = d_model.forward(d_a);
    let d_batch_out = d_model.forward(d_batch);

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&batch_out.data(), &d_batch_out.as_vec());
}

#[test]
fn test_rms_norm() {
    let mut rng = StdRng::seed_from_u64(0);
    // Test single and batch, unoptimized and optimized
    let inp_data = random_vec_rng(15 * 32, &mut rng);
    let weight_data = random_vec_rng(32, &mut rng);
    let mut cx = Graph::new();
    let a = cx.tensor((15, 32)).set(inp_data.clone());

    let model = LayerNorm::new(32, true, false, false, 1e-5, &mut cx);
    model.weight.unwrap().set(weight_data.clone());
    let mut b = model.forward(a).retrieve();

    let _ = cx.compile(<(GenericCompiler, CairoCompiler)>::default(), &mut b);
    cx.execute();

    // Test against dfdx
    let dev = Cpu::default();
    let weight = dev
        .tensor_from_vec(weight_data, (DConst::<32>,))
        .to_dtype::<f32>();
    let a = dev
        .tensor_from_vec(inp_data, (DConst::<15>, DConst::<32>))
        .to_dtype::<f32>();
    let var_f32 = a.clone().square().mean::<_, DAxis<1>>();
    let std_f32 = (var_f32 + 1e-6).sqrt();
    let x_f32 = a / std_f32.broadcast();
    let out = weight.broadcast() * x_f32.to_dtype::<f32>();

    assert_close(&b.data(), &out.to_dtype::<f32>().as_vec());
}

#[test]
fn test_layer_norm() {
    let mut cx = Graph::new();
    let a_data = random_vec(3 * 4 * 5);
    let a = cx.tensor((3, 4, 5)).set(a_data.clone());
    let mut b = a.layer_norm(0, 1e-5).retrieve();
    let mut c = a.layer_norm(2, 1e-5).retrieve();
    let _ = cx.compile(
        <(GenericCompiler, CairoCompiler)>::default(),
        (&mut b, &mut c),
    );
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<3>, DConst::<4>, DConst::<5>));
    let d_b = d_a.clone().normalize::<DAxis<0>>(1e-5);
    let d_c = d_a.normalize::<DAxis<2>>(1e-5);

    assert_close_precision(&b.data(), &d_b.as_vec(), 1e-2);
    assert_close_precision(&c.data(), &d_c.as_vec(), 1e-2);
}
