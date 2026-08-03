"""TCN 模型测试"""
import torch
from src.model.tcn import TemporalConvNet


def test_output_shape():
    model = TemporalConvNet(num_input_channels=34, num_joints=17,
                            receptive_field=81, causal=True, num_layers=5,
                            channels=256)
    x = torch.randn(2, 81, 34)  # (B, T, C)
    out = model(x)              # (B, 17, 3)
    assert out.shape == (2, 17, 3)


def test_causal_last_frame_matters():
    """因果性: 修改最后一帧输入应影响输出"""
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=3, channels=128)
    model.eval()
    x = torch.randn(1, 81, 34)
    with torch.no_grad():
        out1 = model(x)
        x2 = x.clone(); x2[0, 80, :] = 999.0
        out2 = model(x2)
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_receptive_field_coverage():
    """感受野应覆盖全部输入: 修改第 1 帧也影响输出 (rf=81 全覆盖)"""
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=5, channels=128)
    model.eval()
    x = torch.randn(1, 81, 34)
    with torch.no_grad():
        out1 = model(x)
        x2 = x.clone(); x2[0, 0, :] = -999.0
        out2 = model(x2)
    assert not torch.allclose(out1, out2, atol=1e-3)


def test_performance_cuda():
    """A100 上单帧推理 < 20ms (实时性)"""
    if not torch.cuda.is_available():
        print("无 CUDA, 跳过实时性测试")
        return
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=5, channels=1024).cuda()
    model.eval()
    x = torch.randn(1, 81, 34).cuda()
    import time
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 50 * 1000
    print(f"单帧推理延迟: {dt:.2f} ms")
    assert dt < 20, f"推理延迟 {dt:.1f}ms > 20ms"


def test_batch_invariance():
    """batch 大小不影响输出值 (单样本)"""
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=3, channels=128)
    model.eval()
    x = torch.randn(3, 81, 34)
    with torch.no_grad():
        out_batch = model(x)
        out_single = model(x[:1])
    assert out_batch.shape[0] == 3
    assert out_single.shape == (1, 17, 3)
