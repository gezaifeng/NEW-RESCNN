<img width="1323" height="46" alt="image" src="https://github.com/user-attachments/assets/51388279-fe1d-4aa6-96e5-3c2279392bc2" /># NEW-RESCNN
训练策略优化：
1.学习率调度改为自适应 (ReduceLROnPlateau) + 早停
实现方法:
在 train模块中用 ReduceLROnPlateau 替代固定的 StepLR。
监控验证集 loss，如果若干 epoch 内没有下降，就自动把 lr 减半。同时加早停
（patience），避免过拟合和无效训练。
阅读相关文献发现固定步长调度 (StepLR) 在小数据集里不合适：有时候验证 loss 早就进入平台期，但 lr 还没降 → 模型“原地打转”。自适应调度能让 lr 在需要时立即下降，探索更精细的参数空间。早停则能避免跑太久导致过拟合。这样可以显著提升模型在不同溶液浓度下的拟合稳定性。 

△ 2. 在损失里增加“基线”和“形状”正则
实现方法

DC（直流分量）约束基线，TV（一阶差分）约束光谱平滑性。
训练发现，在不同浓度时，吸光度主要表现为整体抬升/下降。
但MSE 更关注峰值，容易放过这种“基线漂移”。
DC 项直接惩罚整体均值差，保证基线稳定。
TV 项抑制波段间的高频噪声，让光谱曲线更符合物理规律（连续、平滑）。这两项约束能极大改善“不同浓度时基线不准、尾部失真”的问题。
<img width="1314" height="441" alt="image" src="https://github.com/user-attachments/assets/bc14a2a0-4af2-445e-8eea-6bce70ed2088" />
