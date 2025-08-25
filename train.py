import math
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import ResCNN
from dataset import SpectralDataset
from WeightedMSEloss import WeightedMSELoss, build_peak_weight

# ========================= 参数配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "D:\Desktop\dataset\guasnoise1-4080\\train"
val_dir   = "D:\Desktop\dataset\guasnoise1-4080\\val"
save_path = "/best_model.pth"
log_path  = "D:\Desktop\model\Res spectrum predict\\training_log.csv"
fig_path  = "D:\Desktop\model\Res spectrum predict\loss_curve.png"

num_epochs = 200
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4

# ========================= 数据加载（与推理保持一致：不做光谱归一化） =========================
train_dataset = SpectralDataset(train_dir, normalize_spectra=False)
val_dataset   = SpectralDataset(val_dir,   normalize_spectra=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ========================= 模型、优化器、调度器 =========================
model = ResCNN().to(device)
core_weight = build_peak_weight(train_dataset, output_dim=76, alpha=3.0).to(device)
criterion_core = nn.MSELoss()
"""WeightedMSELoss(weight=core_weight)"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ======== 额外正则：基线(DC)与一阶平滑(TV) ========
def tv_loss(y):
    return ((y[:,1:] - y[:,:-1])**2).mean()

def dc_baseline_loss(pred, target):
    return (pred.mean(dim=1) - target.mean(dim=1)).pow(2).mean()

lambda_tv = 0.02
lambda_dc = 0.10
lambda_l2_base = 1e-4   # 基线系数已内化在模型里，无法直接取；此处保留钩子以便未来扩展

# ========================= 日志记录变量 =========================
log = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": []}
best_val = float("inf")
patience = 20
wait = 0

# ========================= 训练循环 =========================
for epoch in range(1, num_epochs+1):
    model.train()
    total = 0.0
    for images, spectra in train_loader:
        images = images.to(device).float()
        spectra = spectra.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)

        loss_core = criterion_core(outputs, spectra)
        loss_tv   = tv_loss(outputs)
        loss_dc   = dc_baseline_loss(outputs, spectra)
        loss = loss_core + lambda_tv*loss_tv + lambda_dc*loss_dc
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_loss = total / len(train_loader)

    # ===== 验证 =====
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for images, spectra in val_loader:
            images = images.to(device).float()
            spectra = spectra.to(device).float()
            outputs = model(images)
            loss_core = criterion_core(outputs, spectra)
            loss_tv   = tv_loss(outputs)
            loss_dc   = dc_baseline_loss(outputs, spectra)
            val_total += (loss_core + lambda_tv*loss_tv + lambda_dc*loss_dc).item()

    val_loss = val_total / len(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # ===== 日志 =====
    log["epoch"].append(epoch)
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    log["learning_rate"].append(current_lr)

    print(f"Epoch [{epoch:03d}/{num_epochs}] Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.6f}")

    # ===== 保存最优 & 调度 & 早停 =====
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), save_path)
        print(f"✅ Save model @ epoch {epoch} (val={val_loss:.6f})")
    else:
        wait += 1

    scheduler.step(val_loss)
    if wait >= patience:
        print("⏹️ Early stop triggered.")
        break

# ========================= 保存日志和图像 =========================
df = pd.DataFrame(log)
df.to_csv(log_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
plt.grid(True); plt.legend()
txt = (f"Epochs: {num_epochs}\nBatch: {batch_size}\nLR: {learning_rate}\nWD: {weight_decay}")
plt.gcf().text(0.98, 0.5, txt, fontsize=10, va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout(); plt.savefig(fig_path); plt.close()

print(f"📑 日志: {log_path}")
print(f"📈 曲线: {fig_path}")
print("🎯 训练完成！")
