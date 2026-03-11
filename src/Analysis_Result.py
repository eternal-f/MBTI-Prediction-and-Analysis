import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

df = pd.read_csv('../Output/final_predictions.csv')

# 计算整体准确率
accuracy = accuracy_score(df['label'], df['predicted_label'])
print(f"\n整体预测准确率：{accuracy:.4f}")

# 混淆矩阵
labels = sorted(df['label'].unique())
cm = confusion_matrix(df['label'], df['predicted_label'], labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: True vs Predicted')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('../Plot/confusion_matrix.png')
plt.show()

# 混淆矩阵热力图
labels = sorted(df['label'].unique())
cm = confusion_matrix(df['label'], df['predicted_label'], labels=labels)
plt.figure(figsize=(14, 12))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Reds',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Ratio'},
            annot_kws={'size': 8})
plt.title('Confusion Matrix Heatmap: True vs Predicted (Normalized)', fontsize=14, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../Plot/confusion_matrix_heatmap.png')
plt.show()

# 预测正确的样本数量
df['correct'] = df['label'] == df['predicted_label']
correct_count = df['correct'].sum()
print(f"\n预测正确的样本数量：{correct_count} / {len(df)}")

# 每个类别的准确率
category_accuracy = df.groupby('label')['correct'].mean().sort_values(ascending=False)

# 可视化类别准确率
plt.figure(figsize=(12, 6))
category_accuracy.plot(kind='bar', color='skyblue')
plt.title('Accuracy per MBTI Type')
plt.ylabel('Accuracy')
plt.xlabel('MBTI Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Plot/Accuracy_per_MBTI_type.png')
plt.show()

# 每个类别的错误率
df['error'] = df['label'] != df['predicted_label']
category_error = df.groupby('label')['error'].mean().sort_values(ascending=False)

# 可视化类别错误率
plt.figure(figsize=(14, 6))
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(category_error)))
bars = plt.bar(category_error.index, category_error.values, color=colors, alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.title('Error Rate per MBTI Type', fontsize=14, pad=20)
plt.ylabel('Error Rate', fontsize=12)
plt.xlabel('MBTI Type', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, category_error.max() * 1.15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../Plot/Error_Rate_per_MBTI_type.png')
plt.show()
