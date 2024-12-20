import re
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

red_dict = {
    'Red': ['红',
            '丹', 
            '朱', 
            '赤',
            '绛',
            '胭脂',
            '茜',
            '猩',
            '丹砂',
            '血色',
            '紫绛',
            '玫瑰',
            '绒',
            '春色',
            '荔色',
            '杨妃色']
}

green_dict = {
    'Green':['绿',
             '翠',
             '碧']
}

yellow_dict = {
    'Yellow': ['黄',
               '秋香色',
               '松花色',
               '土色',
               '蜜合色']
}

blue_dict = {
    'Blue': ['青',
             '月白',
             '玉色',
             '蓝',
             '雨过天晴']
}

purple_dict = {
    'purple':['紫',
              '藕合色',
              '茄色',
              '酱色']
}

black_dict = {
    'Black': ['黑',
              '缙',
              '玄',
              '墨',
              '皂']
}

white_dict = {
    'white': ['白']
}

exclude_keywords = ['小红', '黛玉', '怡红院', '红楼梦', '紫鹃', '冯紫英', '翠墨', '墨雨', '明白', '青州', '青儿', '贾蓝', '翠缕', '碧月', '碧痕', '白跑', '白来', '白打', '白日', '白放', '白打', '金钏儿', '金荣', '金陵', '金桂']

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

# 对每个颜色类别提取上下文，并保存为独立的CSV文件
def save_color_context(color_dict, color_name):
    global context  # 使用全局context
    context = []  # 每次提取之前清空context列表
    for color_group, colors in color_dict.items():
        extract_context(text, colors)
    df = pd.DataFrame(context, columns=['color', 'cleaned_context'])
    df.to_csv(f'{color_name}_context.csv', index=False, encoding='utf-8')
    print(f"saved {color_name}_context.csv")

# 提取并保存每个色彩类别的上下文
save_color_context(red_dict, 'red')
save_color_context(green_dict, 'green')
save_color_context(yellow_dict, 'yellow')
save_color_context(blue_dict, 'blue')
save_color_context(purple_dict, 'purple')
save_color_context(black_dict, 'black')
save_color_context(white_dict, 'white')

# 读取所有CSV文件并合并为一个DataFrame
file_names = ['red_context.csv', 'green_context.csv', 'yellow_context.csv', 
              'blue_context.csv', 'purple_context.csv', 'black_context.csv', 'white_context.csv']
df_all = pd.concat([pd.read_csv(file, encoding='utf-8') for file in file_names], ignore_index=True)

# 确保数据被正确加载
print(f"Total rows after merging: {len(df_all)}")

# BERT模型准备
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
embeddings = [get_bert_embeddings(text).numpy().flatten() for text in df_all['cleaned_context']]

# 将 embeddings 转换为 NumPy 数组
embeddings = np.array(embeddings)

# 使用余弦相似度进行 K-means 聚类
similarity_matrix = cosine_similarity(embeddings)
kmeans = KMeans(n_clusters=6, random_state=42)  # 聚类数为6
df_all['cluster'] = kmeans.fit_predict(similarity_matrix)

# 输出每个聚类的样本数量
cluster_counts = df_all['cluster'].value_counts()
print("Cluster counts:\n", cluster_counts)

# PCA降维
pca = PCA(n_components=2)
pca_components = pca.fit_transform(embeddings)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(embeddings)

# 可视化PCA降维结果
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df_all['cluster'], cmap='viridis', marker='o', s=50)
plt.title('PCA Visualization of Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster ID')
plt.show()

# 可视化t-SNE降维结果
plt.figure(figsize=(10, 6))
plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=df_all['cluster'], cmap='viridis', marker='o', s=50)
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster ID')
plt.show()

# 提取每个聚类的代表性词汇
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_all['cleaned_context'])

for cluster_id in range(6):  # 聚类数为6
    cluster_texts = df_all[df_all['cluster'] == cluster_id]['cleaned_context']
    cluster_matrix = vectorizer.transform(cluster_texts)
    terms = vectorizer.get_feature_names_out()
    sum_terms = cluster_matrix.sum(axis=0).A1
    term_frequencies = dict(zip(terms, sum_terms))
    sorted_terms = sorted(term_frequencies.items(), key=lambda item: item[1], reverse=True)

    print(f"Cluster {cluster_id} - Top terms:")
    print(sorted_terms[:20])  # 显示前30个词汇
    print("\n")

# 保存聚类结果到CSV文件
df_all.to_csv('all_colors_context_with_clusters.csv', index=False, encoding='utf-8')




import pandas as pd
import matplotlib.pyplot as plt

# 定义 CSV 文件名
csv_files = ['red_context.csv', 'green_context.csv', 'yellow_context.csv', 
             'blue_context.csv', 'purple_context.csv', 'black_context.csv', 'white_context.csv']

# 统计每个 CSV 文件的行数
file_sizes = {}
for file in csv_files:
    df = pd.read_csv(file, encoding='utf-8')
    file_sizes[file] = len(df)

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(file_sizes.values(), labels=file_sizes.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# 添加标题
plt.title('Number Distribution')

# 显示饼图
plt.show()

# 计算每个聚类的样本数量
cluster_counts = df_all['cluster'].value_counts()

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

# 添加标题
plt.title('Cluster Distribution')

# 显示饼图
plt.show()