import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和图形样式
sns.set_style("whitegrid")
# 读取数据
df = pd.read_csv('../Output/training_metrics.csv')

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Analysis', fontsize=16, fontweight='bold')

# 训练损失和验证损失趋势
axes[0, 0].plot(df['Epoch'], df['Train_Loss'], 'b-', linewidth=2, label='Train_Loss')
axes[0, 0].plot(df['Epoch'], df['Val_Loss'], 'r-', linewidth=2, label='Val_Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Train vs Val')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 验证准确率趋势
axes[0, 1].plot(df['Epoch'], df['Val_Acc'], 'g-', linewidth=2.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Accuracy Tend')
axes[0, 1].grid(True, alpha=0.3)
# 标记关键点
max_acc_epoch = df.loc[df['Val_Acc'].idxmax(), 'Epoch']
max_acc = df['Val_Acc'].max()
axes[0, 1].axhline(y=max_acc, color='red', linestyle='--', alpha=0.7)
axes[0, 1].text(5, max_acc+1, f'Max_Acc: {max_acc:.2f}%', fontweight='bold')

# 损失与准确率的双Y轴图
ax1 = axes[1, 0]
ax2 = ax1.twinx()

color1 = 'tab:blue'
color2 = 'tab:green'
ax1.plot(df['Epoch'], df['Val_Loss'], color=color1, linewidth=2, label='Val_Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Val_Loss', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2.plot(df['Epoch'], df['Val_Acc'], color=color2, linewidth=2, label='Val_Acc')
ax2.set_ylabel('Val_Acc (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_title('Relation of Loss & Accuracy')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

# 训练进度分析 - 按阶段着色
axes[1, 1].scatter(df['Train_Loss'], df['Val_Acc'], c=df['Epoch'],
                   cmap='viridis', s=50, alpha=0.7)
axes[1, 1].set_xlabel('Train_Loss')
axes[1, 1].set_ylabel('Val_Acc (%)')
axes[1, 1].set_title('Train_Loss vs Val_Acc')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Epoch')

plt.tight_layout()
plt.savefig('../Plot/Analysis_of_Training_1.png')
plt.show()

# 创建额外的性能汇总图表
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# 性能改进分析
improvement_phases = [
    (1, 10, 'Increasing Quickly'),
    (11, 30, 'Improving Stably'),
    (31, 60, 'Detailed Optimizing')
]

colors = ['#ff9999', '#66b3ff', '#99ff99']
for i, (start, end, label) in enumerate(improvement_phases):
    phase_data = df[(df['Epoch'] >= start) & (df['Epoch'] <= end)]
    ax.plot(phase_data['Epoch'], phase_data['Val_Acc'],
            color=colors[i], linewidth=3, label=label, marker='o', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('Val_Acc(%)')
ax.set_title('Analysis by Step')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加关键指标标注
final_acc = df['Val_Acc'].iloc[-1]
final_loss = df['Val_Loss'].iloc[-1]
ax.text(0.02, 0.98, f'Final_Accuracy: {final_acc:.2f}%\nFinal_Loss: {final_loss:.4f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../Plot/Analysis_of_Training_2.png')
plt.show()

# 计算关键统计信息
total_epochs = len(df)
final_val_acc = df['Val_Acc'].iloc[-1]
max_val_acc = df['Val_Acc'].max()
max_val_acc_epoch = int(df.loc[df['Val_Acc'].idxmax(), 'Epoch'])
final_train_loss = df['Train_Loss'].iloc[-1]
final_val_loss = df['Val_Loss'].iloc[-1]
train_loss_reduction = df['Train_Loss'].iloc[0] - df['Train_Loss'].iloc[-1]
train_loss_reduction_percent = (train_loss_reduction / df['Train_Loss'].iloc[0] * 100)
val_acc_improvement = df['Val_Acc'].iloc[-1] - df['Val_Acc'].iloc[0]

# 计算收敛速度
threshold_90 = df[df['Val_Acc'] >= 90].iloc[0]['Epoch'] if len(df[df['Val_Acc'] >= 90]) > 0 else None
threshold_95 = df[df['Val_Acc'] >= 95].iloc[0]['Epoch'] if len(df[df['Val_Acc'] >= 95]) > 0 else None

# 保存详细的关键指标到CSV文件
key_metrics = {
    'Metric': [
        'Total_Epochs',
        'Final_Validation_Accuracy',
        'Max_Validation_Accuracy',
        'Epoch_of_Max_Accuracy',
        'Final_Train_Loss',
        'Final_Validation_Loss',
        'Train_Loss_Reduction',
        'Accuracy_Improvement',
        'Epoch_to_90%_Accuracy',
        'Epoch_to_95%_Accuracy'
    ],
    'Value': [
        total_epochs,
        final_val_acc,
        max_val_acc,
        max_val_acc_epoch,
        final_train_loss,
        final_val_loss,
        train_loss_reduction,
        val_acc_improvement,
        threshold_90,
        threshold_95
    ],
    'Unit': [
        'epochs',
        '%',
        '%',
        'epoch',
        '',
        '',
        '',
        '%',
        'epoch',
        'epoch'
    ]
}

key_metrics_df = pd.DataFrame(key_metrics)
key_metrics_df.to_csv('../Output/key_training_metrics.csv', index=False, encoding='utf-8-sig')
