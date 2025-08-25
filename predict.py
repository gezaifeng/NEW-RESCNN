
import torch
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import ResCNN

# 1) 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) 加载模型
model_path = "/best_model.pth"  # 请确认路径
model = ResCNN().to(device)

# 兼容不同保存方式
try:
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
except Exception:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

model.eval()

def visualize_prediction(true_spectrum, predicted_spectrum, wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 760, 5)
    mse = mean_squared_error(true_spectrum, predicted_spectrum)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_spectrum, predicted_spectrum)
    psnr = 10 * math.log10(1 / (rmse ** 2 + 1e-8))
    r2 = r2_score(true_spectrum, predicted_spectrum)

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, true_spectrum, 'bo-', label='real spectrum')
    plt.plot(wavelengths, predicted_spectrum, 'ro-', label='predict spectrum')
    plt.xlabel("wavelength (nm)"); plt.ylabel("absorbance"); plt.title("Spectral prediction")
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nPSNR: {psnr:.2f} dB\nR²: {r2:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(); plt.grid(True); plt.show(block=True)

def robust_block_stat(x, axis=2, q=5):
    # 截断均值：丢弃两端各 q% 的像素，抗眩光/阴影
    low = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100-q, axis=axis, keepdims=True)
    x_clip = np.clip(x, low, high)
    return x_clip.mean(axis=axis)

def predict(rgb_npy_path):
    """
    输入: (4, 6, 100, 3) 的 RGB 样本文件路径
    输出: 模型直接预测的吸光度光谱（不做“反归一化”）
    """
    rgb_data = np.load(rgb_npy_path).astype(np.float32) / 255.0  # (4,6,100,3)

    # 稳健统计（替代简单均值）：(4,6,3)
    rgb_stat = robust_block_stat(rgb_data, axis=2, q=5)

    # 转换为网络输入格式 (3,4,6)
    rgb_input = np.transpose(rgb_stat, (2, 0, 1))
    rgb_tensor = torch.from_numpy(rgb_input).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted = model(rgb_tensor).cpu().numpy().flatten()

    return predicted

if __name__ == "__main__":
    # 示例：请自行修改路径
    test_rgb_path = "D:/Desktop/dataset/guasnoise1-1020/val/rgb_0006.npy"
    true_path     = "D:/Desktop/dataset/guasnoise1-1020/val/spectral_0006.npy"

    pred = predict(test_rgb_path)
    print("预测（前10个点）:", np.round(pred[:10], 4))

    try:
        gt = np.load(true_path)
        visualize_prediction(gt, pred)
        # 另存为 Excel（可选）
        wavelengths = np.arange(380, 760, 5)
        df = pd.DataFrame({"Wavelength (nm)": wavelengths, "Predicted": pred, "True": gt})
        out_xlsx = "spectrum_prediction.xlsx"
        df.to_excel(out_xlsx, index=False)
        print(f"✅ 已保存: {out_xlsx}")
    except Exception as e:
        print("未找到或无法读取真值文件，仅打印预测；错误：", e)
