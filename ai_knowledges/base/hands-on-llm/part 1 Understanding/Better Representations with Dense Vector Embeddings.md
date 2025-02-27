# Word2Vec 模型详解

## 基本概念

嵌入是尝试捕获其含义的数据的向量表示。Word2Vec首先为词汇表中的每个词分配一个随机初始化的向量（在这个例子中是50维的向量）。在训练时，模型会从训练数据中获取成对的词，然后预测这些词是否可能在句子中相邻出现。这是一个非常直观的过程 - 想象我们在阅读一个句子时，某些词经常会一起出现，比如"人工"和"智能"，而有些词很少一起出现，比如"人工"和"香蕉"。通过这个训练过程，Word2Vec能够学习词与词之间的关系，并将这些关系信息提炼到词嵌入向量中。如果两个词经常出现在相似的上下文环境中（即它们倾向于有相同的邻居词），它们的词嵌入向量就会更接近。反之，如果两个词很少出现在相似的上下文中，它们的词嵌入向量就会相距较远。
举个具体的例子： 假设我们有这样的句子："我喜欢吃苹果"和"我喜欢吃香蕉"。通过训练，模型会发现"苹果"和"香蕉"经常出现在相似的上下文中（都跟在"我喜欢吃"后面），因此它们的词向量会比较接近，反映了它们可能属于同类事物（水果）的语义特征。
这种训练方法的优秀之处在于：它不需要人工标注数据，只需要大量的文本语料，就能自动学习到词语之间的语义关系，并将这些关系编码到向量空间中。这也是为什么Word2Vec在发布后迅速成为自然语言处理领域的重要基础模型。 嵌入非常有用，因为它们允许我们测量两个单词之间的语义相似性。使用各种距离度量，我们可以判断一个单词与另一个单词的距离有多近。


Word2Vec 是由 Google 团队在2013年提出的一种高效词嵌入模型，用于将自然语言中的词转换为低维稠密向量。该模型基于分布式假设：上下文相似的词，其语义也相似。Word2Vec 通过神经网络学习词的向量表示，能够捕捉词之间的语义关系和语法特征。

## 模型架构

Word2Vec 主要有两种架构：

### 1. CBOW (Continuous Bag of Words)
- 输入：目标词的上下文词向量
- 输出：预测目标词的概率分布
- 数学表示：
  
  给定上下文词 $c_{1}, c_{2}, ..., c_{2m}$，预测目标词 $w$ 的条件概率：
  
  $p(w|c_{1}, c_{2}, ..., c_{2m}) = \frac{\exp(v'_w \cdot \frac{1}{2m}\sum_{i=1}^{2m}v_{c_i})}{\sum_{w_i \in V}\exp(v'_{w_i} \cdot \frac{1}{2m}\sum_{i=1}^{2m}v_{c_i})}$

### 2. Skip-gram
- 输入：目标词向量
- 输出：预测上下文词的概率分布
- 数学表示：
  
  给定目标词 $w$，预测上下文词 $c$ 的条件概率：
  
  $p(c|w) = \frac{\exp(v'_c \cdot v_w)}{\sum_{c_i \in V}\exp(v'_{c_i} \cdot v_w)}$

## 训练优化

### 1. 负采样 (Negative Sampling)

为解决softmax计算开销大的问题，使用负采样替代：

$\log \sigma(v'_w \cdot v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-v'_{w_i} \cdot v_{w_I})]$

其中：
- $\sigma(x)$ 为sigmoid函数
- $w_I$ 为输入词
- $k$ 为负样本数量
- $P_n(w)$ 为负采样分布

### 2. 层次化Softmax (Hierarchical Softmax)

使用Huffman树结构优化计算：

$p(w|w_I) = \prod_{j=1}^{L(w)-1} \sigma([[n(w,j+1)=ch(n(w,j))]] \cdot v'_{n(w,j)} \cdot v_{w_I})$

其中：
- $L(w)$ 为从根节点到词 $w$ 的路径长度
- $n(w,j)$ 为路径上的第 $j$ 个节点
- $ch(n)$ 为节点 $n$ 的左子节点

## 实现步骤

1. **数据预处理**
   ```python
   def preprocess_text(text):
       # 分词
       tokens = word_tokenize(text.lower())
       # 去除停用词
       tokens = [t for t in tokens if t not in stopwords]
       return tokens
   ```

2. **构建词表和训练数据**
   ```python
   def build_training_data(tokens, window_size):
       # 生成目标词-上下文对
       pairs = []
       for i in range(len(tokens)):
           context = tokens[max(0, i-window_size):i] + 
                     tokens[i+1:min(len(tokens), i+window_size+1)]
           pairs.append((tokens[i], context))
       return pairs
   ```

3. **模型训练**
   ```python
   from gensim.models import Word2Vec
   
   model = Word2Vec(sentences=tokenized_texts, 
                   vector_size=100,    # 词向量维度
                   window=5,           # 上下文窗口大小
                   min_count=1,        # 词频阈值
                   workers=4,          # 训练的线程数
                   sg=1)               # 1 for Skip-gram; 0 for CBOW
   ```

## 词向量特性

### 1. 语义相似性

通过余弦相似度计算词的语义相似度：

$similarity(w_1, w_2) = \frac{v_{w_1} \cdot v_{w_2}}{||v_{w_1}|| \cdot ||v_{w_2}||}$

### 2. 词向量运算

词向量支持基本的代数运算，如：

$v_{King} - v_{Man} + v_{Woman} \approx v_{Queen}$

### 3. 语义聚类

相似语义的词在向量空间中会形成聚类，可用于：
- 同义词发现
- 词义消歧
- 词的类比关系发现

## 应用场景

1. **文本分类**
   - 将词向量作为特征输入分类器
   - 支持迁移学习

2. **情感分析**
   - 捕捉词的情感倾向
   - 构建文档的情感表示

3. **机器翻译**
   - 跨语言词向量映射
   - 零样本翻译

4. **推荐系统**
   - 商品描述的语义匹配
   - 用户行为的语义表示

## 模型评估

### 1. 内在评估
- 词相似度任务
- 词类比任务
- 同义词测试

### 2. 外在评估
- 下游任务性能
  - 文本分类准确率
  - 命名实体识别F1值
  - 机器翻译BLEU分数

## 优化技巧

1. **预训练词向量选择**
   - Google News 预训练向量
   - GloVe 预训练向量
   - FastText 预训练向量

2. **超参数调优**
   - 向量维度：通常选择100-300
   - 窗口大小：根据任务调整2-10
   - 负采样数量：5-20
   - 学习率：初始值0.025，随训练递减

3. **训练技巧**
   - 使用大规模语料
   - 词频截断和二次采样
   - 动态窗口大小
   - 初始学习率退火

## 扩展模型

1. **Doc2Vec**
   - 段落和文档的向量表示
   - 保留词序信息
   - 支持可变长度文本

2. **FastText**
   - 子词嵌入
   - 处理OOV问题
   - 支持多语言

3. **BERT的词嵌入**
   - 上下文相关的词表示
   - 多层双向架构
   - 预训练-微调范式

## 实践注意事项

1. **数据预处理**
   - 充分的文本清洗
   - 合适的分词策略
   - 停用词的处理

2. **训练配置**
   - 适当的词频阈值
   - 合理的向量维度
   - 充足的训练轮数

3. **应用集成**
   - 词向量的标准化
   - OOV词的处理策略
   - 模型的定期更新