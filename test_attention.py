"""
单元测试: Attention.forward (含 KV Cache + 形状注释验证)
运行方式: python test_attention.py
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from llm_LLaMA2 import ModelConfig, Attention, precompute_freqs_cis


def test_attention_training_flash():
    """测试训练模式 (past_kv=None, flash=True, seq_len=16 > 1)"""
    print("=" * 70)
    print("Test 1: 训练模式 — flash=True, seq_len=16")

    args = ModelConfig(dim=512, n_heads=16, n_kv_heads=8, max_seq_len=64, flash_attn=True, dropout=0.0)
    attn = Attention(args)
    attn.eval()

    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)

    b, s = 2, 16
    x = torch.randn(b, s, args.dim)
    output, present_kv = attn(x, freqs_cos, freqs_sin)

    assert output.shape == (b, s, args.dim), f"output shape mismatch: {output.shape}"
    assert isinstance(present_kv, tuple) and len(present_kv) == 2
    k, v = present_kv
    expected_kv_shape = (b, args.n_kv_heads, s, args.dim // args.n_heads)
    assert k.shape == expected_kv_shape, f"k shape mismatch: {k.shape} != {expected_kv_shape}"
    assert v.shape == expected_kv_shape, f"v shape mismatch: {v.shape} != {expected_kv_shape}"
    print(f"  ✅ passed — output:{list(output.shape)}  k:{list(k.shape)}  v:{list(v.shape)}")


def test_attention_training_manual():
    """测试训练模式 (past_kv=None, flash=False, 走手动 attention)"""
    print("=" * 70)
    print("Test 2: 训练模式 — flash=False (手动 attention), seq_len=8")

    args = ModelConfig(dim=512, n_heads=16, n_kv_heads=8, max_seq_len=64, flash_attn=False, dropout=0.0)
    attn = Attention(args)
    attn.eval()

    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)

    b, s = 2, 8
    x = torch.randn(b, s, args.dim)
    output, present_kv = attn(x, freqs_cos, freqs_sin)

    assert output.shape == (b, s, args.dim), f"output shape mismatch: {output.shape}"
    k, v = present_kv
    expected_kv_shape = (b, args.n_kv_heads, s, args.dim // args.n_heads)
    assert k.shape == expected_kv_shape, f"k shape mismatch: {k.shape}"
    assert v.shape == expected_kv_shape, f"v shape mismatch: {v.shape}"
    print(f"  ✅ passed — output:{list(output.shape)}  k:{list(k.shape)}  v:{list(v.shape)}")


def test_attention_inference_kv_cache():
    """测试推理模式: 第1步喂全 prompt, 第2步起每次喂1个 token 带 cache"""
    print("=" * 70)
    print("Test 3: KV Cache 推理 — prompt=8 tokens, 逐步生成 4 tokens")

    args = ModelConfig(dim=512, n_heads=16, n_kv_heads=8, max_seq_len=64, flash_attn=True, dropout=0.0)
    attn = Attention(args)
    attn.eval()

    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)

    b = 1
    prompt_len = 8
    gen_len = 4

    # ——— Step 1: 喂完整 prompt (seq_len=prompt_len, past_kv=None) ———
    x_prompt = torch.randn(b, prompt_len, args.dim)
    out1, past1 = attn(x_prompt, freqs_cos, freqs_sin)
    assert out1.shape == (b, prompt_len, args.dim)
    k1, v1 = past1
    assert k1.shape == (b, args.n_kv_heads, prompt_len, args.dim // args.n_heads)
    assert v1.shape == (b, args.n_kv_heads, prompt_len, args.dim // args.n_heads)

    # ——— Step 2~N: 每次喂 1 token + 传入 cache ———
    past = past1
    total_len = prompt_len
    for i in range(gen_len):
        new_token = torch.randn(b, 1, args.dim)
        out, present = attn(new_token, freqs_cos, freqs_sin, past_kv=past)
        total_len += 1
        assert out.shape == (b, 1, args.dim), f"step {i}: output shape {out.shape}"
        k, v = present
        assert k.shape == (b, args.n_kv_heads, total_len, args.dim // args.n_heads), \
            f"step {i}: k shape {k.shape}, expected total={total_len}"
        assert v.shape == (b, args.n_kv_heads, total_len, args.dim // args.n_heads), \
            f"step {i}: v shape {v.shape}, expected total={total_len}"
        past = present

    print(f"  ✅ passed — prompt={prompt_len} tokens, 生成 {gen_len} tokens 全部通过")
    print(f"      最后 k shape: {list(k.shape)}, v shape: {list(v.shape)}")


if __name__ == "__main__":
    test_attention_training_flash()
    test_attention_training_manual()
    test_attention_inference_kv_cache()
    print("\n" + "=" * 70)
    print("🎉 全部 3 个测试通过！Attention.forward 改动正确。")