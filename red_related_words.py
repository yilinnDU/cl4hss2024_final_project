import re
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 定义红色相关的词汇
red_dict = {
    'Red': ['红', '丹', '朱', '赤', '绛', '胭脂', '茜', '猩', '丹砂', '血色', '紫绛', '玫瑰', '绒', '春色', '荔色', '杨妃色']
}

# 定义需要排除的名词
exclude_keywords = ['小红', '黛玉', '怡红院', '红楼梦']

# 读取《红楼梦》的文本
with open('hongloumeng.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # 清除标点符号
    text = re.sub(r'[^\w\s]', '', text)

# 存储提取的上下文
context = []

# 定义一个函数检查上下文是否包含排除词汇
def contains_exclude_keywords(text):
    return any(keyword in text for keyword in exclude_keywords)

# 定义提取上下文的函数
def extract_context(text, color_list, max_context=10):
    for color in color_list:
        matches = re.finditer(r'.{0,' + str(max_context) + '}' + re.escape(color) + r'.{0,' + str(max_context) + '}', text)
        for match in matches:
            context_text = match.group()
            # 过滤掉包含排除词汇的上下文
            if not contains_exclude_keywords(context_text):
                context.append((color, context_text))

# 对每个颜色词汇提取上下文
for color_group, colors in red_dict.items():
    extract_context(text, colors)

# 将提取的上下文保存到DataFrame
df = pd.DataFrame(context, columns=['color', 'cleaned_context'])

# 保存到CSV文件
df.to_csv('red_context.csv', index=False, encoding='utf-8')
print("saved red_context.csv")

# 清理'cleaned_context'列，去除不必要的字符（比如空格、标点符号等）
df['cleaned_context'] = df['cleaned_context'].apply(lambda x: re.sub(r'[^\w\s]', '', x).strip())

# 打印一下清理后的数据
print(df[['cleaned_context']].head())  # 这里输出'cleaned_context'列

# BERT 模型准备
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 获取BERT嵌入的函数
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 使用最后一层的平均作为句子嵌入
    return embeddings

# 获取每个上下文的BERT嵌入
embeddings = [get_bert_embeddings(text).numpy().flatten() for text in df['cleaned_context']]

# 进行K-means聚类
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(np.array(embeddings))

# 输出结果
print(df[['cleaned_context', 'cluster']].head())  # 这里输出'cleaned_context'和'cluster'列

# 保存带有聚类结果的CSV文件
df.to_csv('red_context_with_clusters.csv', index=False, encoding='utf-8')

# 可视化聚类结果
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(np.array(embeddings))

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50)
plt.colorbar()
plt.title("PCA of BERT Embeddings - Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# 使用t-SNE进行降维到二维
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(np.array(embeddings))

plt.figure(figsize=(10, 6))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=df['cluster'], cmap='viridis', s=50)
plt.colorbar()
plt.title("t-SNE of BERT Embeddings - Clusters")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# 统计每个聚类的样本数量
cluster_counts = df['cluster'].value_counts()
print(cluster_counts)

# 提取每个聚类的代表性词汇
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_context'])

for cluster_id in range(df['cluster'].nunique()):
    cluster_texts = df[df['cluster'] == cluster_id]['cleaned_context']
    cluster_matrix = vectorizer.transform(cluster_texts)
    terms = vectorizer.get_feature_names_out()
    sum_terms = cluster_matrix.sum(axis=0).A1
    term_frequencies = dict(zip(terms, sum_terms))
    sorted_terms = sorted(term_frequencies.items(), key=lambda item: item[1], reverse=True)
    
    print(f"Cluster {cluster_id} - Top terms:")
    print(sorted_terms[:20])  # 显示前10个词汇
    print("\n")


import matplotlib.pyplot as plt
cluster_counts = df['cluster'].value_counts()

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# 添加标题
plt.title('Cluster Distribution')

# 显示饼图
plt.show()