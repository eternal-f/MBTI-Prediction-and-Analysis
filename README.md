# MBTI Personality Prediction & Analysis  
This project aims to predict MBTI personality types from Twitter text using deep learning and linguistic analysis. It combines exploratory data analysis, lexical feature extraction, and a TextCNN-based classification model to uncover linguistic patterns associated with different personality dimensions.  

# Project Overview  
**Goal**: Predict MBTI personality types from user-generated Twitter content.  
**Dataset**: Kaggle - MBTI Personality Type Twitter Dataset  
**Approach**:
Data cleaning and preprocessing  
Exploratory analysis (word clouds, keyness analysis, sentiment trends)  
TextCNN model for multi-class classification (16 personality types)  
Model evaluation using accuracy, loss curves, and confusion matrix  

# Key Contributions  
**Data Preprocessing**: Removed noise (URLs, mentions, emojis), filtered non-English comments, and balanced dataset across types.  
**Modeling**: Implemented a TextCNN architecture with word embeddings, convolutional layers, and fully connected output.  
**Evaluation**: Achieved ~80% accuracy on filtered data; analyzed model limitations and overfitting risks.  
**Linguistic Insight**: Identified distinctive keywords for each MBTI dimension (e.g., "raffle" for Perceiving, "artwork" for Feeling).  

# Repository Structure  
Main  
├── src/  
│   ├── Dataclean.py -------------   Data cleaning and preprocessing  
│   ├── Dataselect.py ------------   Train-test split (80:20)  
│   ├── Train.py -----------------   Model training and vocabulary building  
│   ├── Predict.py ---------------   Evaluation on test set  
│   ├── Demo.py ------------------   Interactive prediction demo  
│   ├── Analysis_Result.py -------   Post-training result analysis  
│   └── Analysis_Training.py -----   Training loss visualization  
│  
├── model/  
│   ├── personality_model.pth ----   Trained TextCNN model    
│   └── vocab.pth ----------------   Vocabulary used for tokenization  
│  
├── demo/  
│   ├── Analysis_of_Training_2.png ---   Training loss curve  
│   ├── confusion_matrix_heatmap.png -   Confusion matrix  
│   └── Demo_video.mp4 ---------------   Demo video of prediction  
│  
├── data/ ------------   Preprocessed train/test data  
├── Report/ ----------   Project report (PDF)  
└── README.md  

# Model Architecture: TextCNN
The model consists of:  
**Embedding Layer**: Converts tokens into dense vector representations.  
**Convolutional Layers**: Multiple filters (e.g., 2, 3, 4) capture n-gram features.    
**Pooling & Output**: Max-pooling + fully connected layers for final classification.  
<img width="416" height="217" alt="image" src="https://github.com/user-attachments/assets/c4923851-6b1b-41b4-8690-61a814d17186" />  

# Results
**Accuracy**: ~80% on clean, filtered test data  
**Sensitivity**: Performance drops significantly (~10%) on noisy inputs  
**Overfitting**: Observed during training; mitigated with data balancing and regularization  
<img width="1000" height="600" alt="Analysis_of_Training_2" src="https://github.com/user-attachments/assets/340b57e9-8a1e-46ad-a878-8cf0f7c07c43" />

# MBTI 性格类型预测与分析  
本项目旨在通过深度学习和语言分析，利用推特文本数据预测用户的MBTI性格类型。项目结合了探索性数据分析、词汇特征提取以及基于TextCNN的分类模型，以揭示与不同性格维度相关的语言模式。  
  
# 项目概述  
**目标**：根据用户生成的推特内容预测其MBTI性格类型。 
**数据集**：Kaggle - MBTI Personality Type Twitter Dataset  
**方法**：  
数据清洗与预处理  
探索性分析（词云、关键词分析、情感趋势）  
基于TextCNN的多分类模型（16种性格类型）  
通过准确率、损失曲线和混淆矩阵进行模型评估  
  
# 主要贡献
**数据预处理**：去除噪声（如URL、提及、表情符号），过滤非英文评论，并在各类型间平衡数据集。  
**模型构建**：实现了一个基于TextCNN的架构，包含词嵌入层、卷积层和全连接输出层。  
**模型评估**：在过滤后的数据上达到约80%的准确率；分析了模型局限性及过拟合风险。  
**语言分析**：识别出与各MBTI维度相关的特征词（例如，“raffle”与感知型相关，“artwork”与情感型相关）。  

# 文件结构
Main  
├── src/  
│ ├── Dataclean.py ------------- 数据清洗与预处理  
│ ├── Dataselect.py ------------ 训练集与测试集划分（80:20）  
│ ├── Train.py ----------------- 模型训练与词汇表构建  
│ ├── Predict.py --------------- 在测试集上进行评估  
│ ├── Demo.py ------------------ 交互式预测演示  
│ ├── Analysis_Result.py ------- 训练后结果分析  
│ └── Analysis_Training.py ----- 训练损失可视化  
│  
├── model/  
│ ├── personality_model.pth ---- 训练好的TextCNN模型  
│ └── vocab.pth ---------------- 用于分词的词汇表  
│  
├── demo/  
│ ├── Analysis_of_Training_2.png --- 训练损失曲线图  
│ ├── confusion_matrix_heatmap.png - 混淆矩阵热力图  
│ └── Demo_video.mp4 --------------- 预测演示视频  
│  
├── data/ ------------ 预处理后的训练/测试数据  
├── Report/ ---------- 项目报告（PDF）  
└── README.md  
  
# 模型架构：TextCNN  
模型由以下部分组成：  
**嵌入层**：将分词后的词元转换为密集向量表示。  
**词卷积层**：使用多种尺寸的卷积核（例如2、3、4）来捕捉n-gram特征。  
**池化与输出层**：采用最大池化，并通过全连接层进行最终分类。  
<img width="831" height="433" alt="image" src="https://github.com/user-attachments/assets/31f96462-116b-409b-877b-38d0adabcaf4" />


# 实验结果  
**准确率**：在干净、过滤后的测试数据上达到约80%。  
**鲁棒性**：在含有噪声的输入上，性能显著下降（约10%）。  
**过拟合**：训练过程中出现过拟合现象，通过数据平衡和正则化技术得到缓解。  
<img width="1000" height="600" alt="Analysis_of_Training_2" src="https://github.com/user-attachments/assets/a4684602-fca9-46a5-88dd-443c8d4ab1ef" />

