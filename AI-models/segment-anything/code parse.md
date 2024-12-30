# common.py

```python
class MLPBlock(nn.Module):  
    def __init__(  
        self,  
        embedding_dim: int,  
        mlp_dim: int,  
        act: Type[nn.Module] = nn.GELU,  
    ) -> None:  
        super().__init__()  
        # 将输入从 `embedding_dim` 大小映射到 `mlp_dim`
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        # 将中间的 `mlp_dim` 再映射回 `embedding_dim`  
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)  
        self.act = act()  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.lin2(self.act(self.lin1(x)))
```
一个简单经典的`线性 -> 激活 -> 线性`多层感知机模块，默认使用GeLU激活函数
- `embedding_dim`: 输入和输出的特征维度，表示输入张量（或向量）的大小。
- `mlp_dim`: 全连接层的中间隐层维度，通常比 `embedding_dim` 更大，用来提供更高的表达能力。

```python
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa  
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa  
class LayerNorm2d(nn.Module):  
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:  
        super().__init__()  
        self.weight = nn.Parameter(torch.ones(num_channels))  
        self.bias = nn.Parameter(torch.zeros(num_channels))  
        self.eps = eps  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        u = x.mean(1, keepdim=True)  
        s = (x - u).pow(2).mean(1, keepdim=True)  
        x = (x - u) / torch.sqrt(s + self.eps)  
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  
        return x
```
- `weight` 与 `bias`：
    - 大小均为 `[num_channels]`，每个通道都有自己的一组可学习缩放系数和偏置。
    - 在最后一步通过 `[..., None, None]` 形状变换来广播到 `[N, C, H, W]` 中的所有像素上。
- `eps`：防止除零或数值过小导致的数值不稳定。

- `forward`方法中执行了一个典型的前向传播：
	- 均值`u`: 对维度1`channel`求均值，得到每个样本、每个像素位置上的通道平均值。
	- 方差 `s`: 计算 `(x - u)^2` 后在通道维再次求平均，得到每个位置的通道方差。
	- 归一化: 减去通道均值、除以通道标准差，实现“通道归一化”。`self.eps` 避免分母为 0 或数值过小而引发不稳定。
	- 线性变换: `weight` 和 `bias` 的形状是 `[C]`，通过 `[:, None, None]` 进行广播，使之在 `[N, C, H, W]` 的通道维上分别对应每一个通道。
	- `self.weight[:, None, None]`: 这里的 `[:, None, None]` 会把 `self.weight` 从形状 `[C]` 扩展为 `[C, 1, 1]`。
	- 同样地，`self.bias[:, None, None]` 也从 `[C]` 扩展为 `[C, 1, 1]`。

**PyTorch 广播规则**
简单来说，**从右到左**比较维度：
1. 如果两个维度相同，或者其中一个是 1，则可以广播到相同的大小。
2. 如果两个维度不相等，而且都不是 1，则会报错（无法广播）。
在本例中：
- `[C, 1, 1]` 可以被看作 `[1, C, 1, 1]`（在最左边再补一个 1，用于对齐 `N`）。
- 再与 `[N, C, H, W]` 对比：
    - 第 1 维：1 vs N，广播到 N
    - 第 2 维：C vs C，匹配
    - 第 3 维：1 vs H，广播到 H
    - 第 4 维：1 vs W，广播到 W
所以最终得到 `[N, C, H, W]` 的输出
经过广播，每个通道都会乘以与该通道对应的 `weight[c]`，并加上 `bias[c]`。也就是说：
- 对于通道 c，
    - 先将 `x[:, c, :, :]` 这部分数据乘以 `weight[c]`，
    - 再加上 `bias[c]`。
- 这样就实现了“每个通道都有自己的一对缩放和偏置参数”的效果。

如果不使用广播，就必须手动写一个循环，对每个通道 `c` 做类似:
```python
for c in range(C):
    x[:, c, :, :] = x[:, c, :, :] * self.weight[c] + self.bias[c]
```

**与 PyTorch 官方 `LayerNorm` 的区别**
- PyTorch 的 `nn.LayerNorm` 通常可以指定要在张量的哪些维度上进行归一化（比如针对 `[C, H, W]` 做全部归一化）。默认情况下，LayerNorm 会把指定的“特征维度”都视作一个整体去计算均值和方差。
- 这段自定义的 `LayerNorm2d` 则专门对“通道维”进行统计（在维度 1 求均值与方差），相当于对 **每个像素位置** 的所有通道进行归一化。
    - 这种做法从名字上看像 Layer Norm，但就**归一化维度**而言，更像将“通道”视为特征维，固定了只在通道上做统计，而对空间维度 `[H, W]` 不做合并处理。
    - 在一些场景中，这种方式也被称作“Instance Norm”的一种变体（如果只在单个样本和通道上做归一化，并且不跨样本）。

---

# image encoder 

```python
class PatchEmbed(nn.Module):  
    """  
    Image to Patch Embedding.    """  
    def __init__(  
        self,  
        kernel_size: Tuple[int, int] = (16, 16),  
        # 卷积核滑动的步长。默认值 (16, 16)，表示每次滑动16个像素，patches 之间不会有重叠。
        stride: Tuple[int, int] = (16, 16),  
        padding: Tuple[int, int] = (0, 0),  
        in_chans: int = 3,  
        embed_dim: int = 768,  # 每个 patch 映射后的嵌入维度
    ) -> None:  
        """  
        Args:            
	        kernel_size (Tuple): kernel size of the projection layer.           
	        stride (Tuple): stride of the projection layer.            
	        padding (Tuple): padding size of the projection layer.           
	        in_chans (int): Number of input image channels.            
	        embed_dim (int): Patch embedding dimension.        
		"""        
		super().__init__()  
        self.proj = nn.Conv2d(in_chans, 
			embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
			)  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = self.proj(x)  
        # B C H W -> B H W C  
        x = x.permute(0, 2, 3, 1)  # 维度变换
        return x
```

`PatchEmbed` 是一个将图像转换为**Patch Embedding**的模块，广泛应用于**Vision Transformer (ViT)** 等模型中。**Patch Embedding** 是一种将 **图像数据** 转换为 **一系列嵌入向量（embedding tokens）** 的方法。它的核心功能是：
- 将输入的 2D 图像划分为若干个小块（patches）。
- 将每个小块映射到一个特征向量（embedding/Token）。
- 将这些 **Token** 作为 Transformer 的输入，最终输出一个表示这些 patch 的嵌入张量。

 **卷积映射**
- 输入：
    - `x` 的形状是 `[B, C, H, W]`
        - `B`: batch size
        - `C`: 通道数（`in_chans`）
        - `H`: 图像高度
        - `W`: 图像宽度
- 输出（经过 `self.proj`）：
    - `x` 的形状变为 `[B, embed_dim, H_p, W_p]`
        - `embed_dim`: 每个 patch 的嵌入维度
        - `H_p`: patch 的行数（受 stride 和 kernel_size 影响）
        - `W_p`: patch 的列数（受 stride 和 kernel_size 影响）

**为什么要这样调整？**
- 这种排列方式更适合 Transformer 模型的输入格式。
- `[H_p, W_p]` 表示每个位置的 patch。
- `embed_dim` 是每个 patch 的特征向量。
- 每个位置的 patch 都可以看作一个 token（类似于 NLP 中的单词 token）。

假设：
- 输入：`x`，形状 `[1, 3, 224, 224]`（单张 RGB 图像，大小 224x224）。
- `kernel_size=(16,16)`，`stride=(16,16)`，`embed_dim=768`。
 **Step 1: 卷积映射**
- 划分为 22416=14\frac{224}{16} = 1416224​=14 个 patch（行和列各 14 个）。
- 输出形状：`[1, 768, 14, 14]`
 **Step 2: 维度变换**
- 调整形状：`[1, 14, 14, 768]`
这个张量可以看作：
- `14 × 14 = 196` 个 token。
- 每个 token 是一个维度为 `768` 的向量。
- 这与 NLP 模型中的 token embedding 非常类似。


```python
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:  
    """  
    Get relative positional embeddings according to the relative positions of 
    query and key sizes.    
    
    Args:        
	    q_size (int): size of query q. Query（查询特征图）的大小（例如，高度或宽度）。
	    k_size (int): size of key k. Key（键特征图）的大小（例如，高度或宽度）。       
	    rel_pos (Tensor): relative position embeddings (L, C).  
		    - L: 相对位置的总数量（例如，2 × max(q_size, k_size) - 1）
			- C: 每个相对位置的通道数（特征维度）
    Returns:        
	    Extracted positional embeddings according to relative positions.    
	"""    
	max_rel_dist = int(2 * max(q_size, k_size) - 1)  
    # Interpolate rel pos if needed.  
    if rel_pos.shape[0] != max_rel_dist:  
        # Interpolate rel pos.  
        rel_pos_resized = F.interpolate(  
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),  
            size=max_rel_dist,  
            mode="linear",  
        )  
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)  
    else:  
        rel_pos_resized = rel_pos  
  
    # Scale the coords with short length if shapes for q and k are different.  
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)  
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)  
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)  
  
    return rel_pos_resized[relative_coords.long()]
```

- **最大相对位置距离**的数量由公式 2×max(q_size,k_size)−12 \times \text{max}(q\_size, k\_size) - 12×max(q_size,k_size)−1 计算得出。
- 它表示两个特征图之间可能存在的最大相对位置索引。

 **1. 什么是相对位置距离？**

在 Transformer 或注意力机制中，**相对位置编码（Relative Positional Embedding）** 是用来表示 **query (Q)** 和 **key (K)** 之间的**相对位置关系**的。

- **绝对位置编码**：为每个位置提供一个独立的编码。
- **相对位置编码**：表示每对位置之间的**距离关系**，例如：
    - 左边、右边、上面、下面、相隔几个单位等。

 **查询-键的相对位置**

假设：

- `q_size = 3`（查询序列长度）
- `k_size = 3`（键序列长度）

那么 `query` 和 `key` 之间的相对位置差可以表示为：

Δ=i−j\Delta = i - jΔ=i−j

- iii 是 Query 的索引（从 0 到 q_size-1）
- jjj 是 Key 的索引（从 0 到 k_size-1）

所有可能的相对位置差的范围是：

Δ∈[−(k_size−1),+(q_size−1)]\Delta \in [-(k\_size - 1), +(q\_size - 1)]Δ∈[−(k_size−1),+(q_size−1)]

- 最小差值是：−(k_size−1)-(k\_size - 1)−(k_size−1)
- 最大差值是：(q_size−1)(q\_size - 1)(q_size−1)

为了表示所有可能的相对位置，**需要一个足够大的数组来存储这些编码**。

---

**2. 为什么是 2×max⁡(q_size,k_size)−12 \times \max(q\_size, k\_size) - 12×max(q_size,k_size)−1？**

 **(1) 最大相对距离**

- 如果 `q_size` 和 `k_size` 不相等，则最大差距可能是最大长度。
- 举例：
    - `q_size = 5, k_size = 3`
    - 最大正差：5−1=45 - 1 = 45−1=4
    - 最大负差：−(3−1)=−2- (3 - 1) = -2−(3−1)=−2

所以，**最大相对位置距离**需要涵盖从最大负差到最大正差的所有可能值。

 **(2) 为什么是 2 倍最大长度 - 1？**

如果我们考虑所有的相对位置差值的范围：

[−(k_size−1),+(q_size−1)][-(k\_size - 1), +(q\_size - 1)][−(k_size−1),+(q_size−1)]

最大差距是：

- 如果两个维度不同，最大差距会是：

Δmax⁡=max⁡(q_size−1,k_size−1)\Delta_{\max} = \max(q\_size - 1, k\_size - 1)Δmax​=max(q_size−1,k_size−1)

而总的可能相对位置数量是：

Δcount=(q_size−1)+(k_size−1)+1\Delta_{\text{count}} = (q\_size - 1) + (k\_size - 1) + 1Δcount​=(q_size−1)+(k_size−1)+1

将其简化为：

=(q_size+k_size−2)+1=q_size+k_size−1= (q\_size + k\_size - 2) + 1 = q\_size + k\_size - 1=(q_size+k_size−2)+1=q_size+k_size−1

进一步取最大值（确保覆盖更大的一个维度）：

=2×max⁡(q_size,k_size)−1= 2 \times \max(q\_size, k\_size) - 1=2×max(q_size,k_size)−1

### **(3) 直观理解**

- 如果两个维度完全一样（比如 `q_size = k_size = 3`），那么相对位置的范围是 `[-2, -1, 0, 1, 2]`，总共 5 个。
- 如果两个维度不一样，最大范围会受到较大的那个维度的控制，因此需要取 `max`。

---

## 🛠️ **3. 举例验证**

### **例子 1：q_size = 3, k_size = 3**

- 相对位置差的范围是：[−2,−1,0,1,2][-2, -1, 0, 1, 2][−2,−1,0,1,2]
- 总共 5 个位置。
- 2×max⁡(3,3)−1=52 \times \max(3, 3) - 1 = 52×max(3,3)−1=5 ✅

### **例子 2：q_size = 4, k_size = 2**

- 最大正差：4−1=34 - 1 = 34−1=3
- 最大负差：−(2−1)=−1-(2 - 1) = -1−(2−1)=−1
- 相对位置范围是：[−1,0,1,2,3][-1, 0, 1, 2, 3][−1,0,1,2,3]
- 总共 5 个位置。
- 2×max⁡(4,2)−1=72 \times \max(4, 2) - 1 = 72×max(4,2)−1=7 ✅

---

## 📊 **4. 结论**

- 相对位置的范围来自查询（`q`) 和键（`k`）之间的**所有可能的相对位置差**。
- 为了覆盖所有可能的差距，我们需要：

Δcount=2×max⁡(q_size,k_size)−1\Delta_{\text{count}} = 2 \times \max(q\_size, k\_size) - 1Δcount​=2×max(q_size,k_size)−1

- **核心思想**：
    - 每个查询和每个键之间都有一个相对位置差。
    - 这个差的范围是由较大维度决定的。
    - 需要一个长度为 `2 × max(q_size, k_size) - 1` 的相对位置编码数组来存储所有可能的相对位置。

---

## 💡 **5. 直观记忆**

- **最大差距** = 左边最大负数到右边最大正数之间的跨度。
- **2 倍最大维度** = 左边和右边的总跨度。
- **减 1** = 起始点与终点之间的实际差值。