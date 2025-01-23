# 词袋模型 (Bag of Words)

## 基本概念

词袋模型(Bag of Words, BoW)是自然语言处理中最基础的文本表示方法之一。它将文本表示为一个"词袋",忽略词序和语法,仅考虑词频信息。这种方法虽然简单,但在文本分类、情感分析等任务中仍具有重要应用价值。假设我们有两个句子要为其创建数字表示。词袋模型的第一步是标记化，即将句子分割成单个单词或子词（标记）的过程。使用我们的词汇量，我们只需计算每个句子中某个单词出现的频率，实际上就创建了一个词袋。因此，词袋模型旨在以数字形式创建文本表示，同时，词袋模型旨在以数字形式创建文本表示，称为向量或向量表示。 词袋虽然是一种优雅的方法，但也有一个缺陷。它认为语言只不过是几乎字面上的词袋，而忽略了文本的语义本质或意义。

## 数学表示

### 词向量表示

给定文档集合 $D = \{d_1, d_2, ..., d_n\}$ 和词汇表 $V = \{w_1, w_2, ..., w_m\}$,每个文档 $d_i$ 可以表示为一个 $m$ 维向量：

$d_i = [t_{i1}, t_{i2}, ..., t_{im}]$

其中 $t_{ij}$ 表示词 $w_j$ 在文档 $d_i$ 中的权重。

### 常用权重计算方法

1. **布尔权重**
   最简单的表示方法,仅考虑词是否出现：
   
   $t_{ij} = \begin{cases} 1, & \text{if } w_j \text{ appears in } d_i \\ 0, & \text{otherwise} \end{cases}$

2. **词频权重(TF)**
   考虑词在文档中出现的频率：
   
   $t_{ij} = f_{ij}$

   其中 $f_{ij}$ 为词 $w_j$ 在文档 $d_i$ 中的出现次数。

3. **TF-IDF权重**
   结合词频(TF)和逆文档频率(IDF)：
   
   $t_{ij} = tf_{ij} \times idf_j$

   其中：
   - $tf_{ij}$ 为词 $w_j$ 在文档 $d_i$ 中的词频
   - $idf_j = \log\frac{N}{df_j}$
   - $N$ 为文档总数
   - $df_j$ 为包含词 $w_j$ 的文档数

## 实现步骤

1. **分词**
   - 将文本拆分为单词序列
   - 去除停用词(可选)
   - 词形还原或词干提取(可选)

2. **构建词汇表**
   - 统计所有不同词项
   - 为每个词项分配唯一索引
   - 可以设置词频阈值筛选词表(可选)

3. **向量化**
   - 统计每个文档中各词项的权重
   - 构建文档-词项矩阵

## 优缺点分析

### 优点
1. 实现简单,计算高效
2. 适用于基础文本分类任务
3. 与其他模型易于集成
4. 特征直观可解释

### 缺点
1. 忽略词序信息
2. 忽略词的语义关系
3. 高维稀疏性问题
4. 无法处理词的多义性

## 应用场景

1. **文本分类**
   - 垃圾邮件过滤
   - 新闻分类
   - 情感分析

2. **信息检索**
   - 文档相似度计算
   - 关键词提取
   - 文档聚类

3. **特征工程**
   - 作为深度学习模型的输入特征
   - 与其他特征的组合使用

## 代码示例

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 示例文档
documents = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third document.',
]

# 词频向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print("词汇表:", vectorizer.get_feature_names_out())
print("词频矩阵:\n", X.toarray())

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF矩阵:\n", X_tfidf.toarray())
```

## 扩展与改进

1. **N-gram模型**
   - 考虑连续N个词的组合
   - 部分保留词序信息
   - 表达能力更强

2. **主题模型**
   - LSA(潜在语义分析)
   - LDA(潜在狄利克雷分配)
   - 降维并发现潜在主题

3. **词嵌入**
   - Word2Vec
   - GloVe
   - FastText
   - 获取词的分布式表示

## 参考公式

### 余弦相似度计算

用于计算文档向量间的相似度：

$similarity(d_i, d_j) = \frac{d_i \cdot d_j}{||d_i|| \cdot ||d_j||} = \frac{\sum_{k=1}^m t_{ik}t_{jk}}{\sqrt{\sum_{k=1}^m t_{ik}^2} \sqrt{\sum_{k=1}^m t_{jk}^2}}$

### 归一化词频

减少文档长度的影响：

$tf_{ij} = \frac{f_{ij}}{\max_k f_{ik}}$