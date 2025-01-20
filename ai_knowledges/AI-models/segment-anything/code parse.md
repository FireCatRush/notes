# common.py
## MLPBlock
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
## LayerNorm2d
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
## PatchEmbed
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

另外一种patchembed的实现方法是手动拉平+线性映射：
```python
x = x.unfold(2, 16, 16).unfold(3, 16, 16)  # 将图像分成小块
x = x.flatten(2).transpose(1, 2)  # 拉平为序列
x = nn.Linear(3 * 16 * 16, 768)(x)  # 映射到嵌入维度
```

---
## get_rel_pos
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
`get_rel_pos` 是一个用于获取**相对位置编码（Relative Positional Embedding）** 的函数，常用于 Transformer 模型中，特别是在视觉 Transformer（如 ViT）或具有注意力机制的网络中。它的作用是：
- **动态调整**相对位置编码的大小，以适配不同大小的 `query` 和 `key`。
- **返回**与 `query` 和 `key` 相对位置匹配的嵌入向量。
- 
---
**最大相对位置距离**的数量由公式 $2 \times \max(q\_size, k\_size) - 12 \times \max(q\_size, k\_size) - 1$计算得出。它表示两个特征图之间可能存在的最大相对位置索引。

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
$$ \Delta = i - j $$
- $i$ 是 Query 的索引（从 0 到 q_size-1）
- $j$ 是 Key 的索引（从 0 到 k_size-1）
所有可能的相对位置差的范围是：
$$
\Delta \in [-(k\_size - 1), +(q\_size - 1)]
$$
- 最小差值是：$-(k\_size - 1)$
- 最大差值是：$+(q\_size - 1)$
为了表示所有可能的相对位置，**需要一个足够大的数组来存储这些编码**。

**2. 为什么是  $2 \times \max(q\_size, k\_size) - 12 \times \max(q\_size, k\_size) - 1$**
 **(1) 最大相对距离**

- 如果 `q_size` 和 `k_size` 不相等，则最大差距可能是最大长度。
- 举例：
    - `q_size = 5, k_size = 3`
    - 最大正差：5−1=4
    - 最大负差：−(3−1)=−2
所以，**最大相对位置距离**需要涵盖从最大负差到最大正差的所有可能值。

 **(2) 为什么是 2 倍最大长度 - 1？**
如果我们考虑所有的相对位置差值的范围：
$$[-(k\_size - 1), +(q\_size - 1)]$$
最大差距是：
- 如果两个维度不同，最大差距会是：
$$
\Delta_{\max} = \max(q\_size - 1, k\_size - 1)
$$
而总的可能相对位置数量是：
$$
\Delta_{\text{count}} = (q\_size - 1) + (k\_size - 1) + 1
$$
将其简化为：
$$
= (q\_size + k\_size - 2) + 1 = q\_size + k\_size - 1
$$
进一步取最大值（确保覆盖更大的一个维度）：
$$
= 2 \times \max(q\_size, k\_size) - 1
$$
**(3) 直观理解**
- 如果两个维度完全一样（比如 `q_size = k_size = 3`），那么相对位置的范围是 `[-2, -1, 0, 1, 2]`，总共 5 个。
- 如果两个维度不一样，最大范围会受到较大的那个维度的控制，因此需要取 `max`。

 **3. 举例验证**
**例子 1：q_size = 3, k_size = 3**
- 相对位置差的范围是：\[-2, -1, 0, 1, 2\]
- 总共 5 个位置。
- $2 \times \max(3, 3) - 1 = 5$ 

 **例子 2：q_size = 4, k_size = 2**
- 最大正差：4−1=3
- 最大负差：−(2−1)=−1
- 相对位置范围是：\[−1,0,1,2,3\]
- 总共 5 个位置。
- $2 \times \max(4, 2) - 1 = 7$

 **4. 结论**

- 相对位置的范围来自查询（`q`) 和键（`k`）之间的**所有可能的相对位置差**。
- 为了覆盖所有可能的差距，我们需要：
$$\Delta_{\text{count}} = 2 \times \max(q\_size, k\_size) - 1$$
- **核心思想**：
    - 每个查询和每个键之间都有一个相对位置差。
    - 这个差的范围是由较大维度决定的。
    - 需要一个长度为 `2 × max(q_size, k_size) - 1` 的相对位置编码数组来存储所有可能的相对位置。
---
**调整相对位置编码的大小**
在 Transformer 和其他注意力机制中，**相对位置编码（Relative Positional Embedding）** 是为了让模型能够感知序列中不同位置之间的**相对距离**，而不是绝对位置。这种机制对于图像处理特别重要，因为图像的空间关系（像素之间的位置关系）直接影响特征的含义。

**相对位置编码的核心目标**
- **对齐 `query` 和 `key`**
    - 在注意力机制中，`query` 和 `key` 可能来自不同的特征图（例如经过不同步长或不同分辨率的卷积处理后的特征图）。
    - 这导致 `query` 和 `key` 的维度可能不一致（例如，`q_size ≠ k_size`）。
    - 为了保证相对位置编码可以准确地描述两者之间的相对位置关系，**需要对相对位置编码进行调整**。
- **确保覆盖所有可能的相对位置**
    - 如果原始的相对位置编码 `rel_pos` 的长度（L）不匹配当前的 `query` 和 `key` 的大小关系，直接索引会导致**越界错误**或**信息缺失**。
    - 需要将相对位置编码调整到合适的长度，确保每个 `query` 和 `key` 之间的相对位置都能正确映射到 `rel_pos` 中。

 **调整的实际场景**
 **(1) 空间分辨率变化**
在图像 Transformer 中，`query` 和 `key` 可能来自不同的网络层，导致它们的分辨率不同：
- 例如：
    - `query` 来自特征图的分辨率为 14×1414 \times 1414×14
    - `key` 来自特征图的分辨率为 28×2828 \times 2828×28
- 原始的相对位置编码可能是为某个特定的分辨率设计的（例如，`max_rel_dist = 27`）。
- 如果直接将这个相对位置编码映射到不同分辨率的 `query` 和 `key` 上，可能会导致索引不匹配或覆盖不全。
**(2) 共享相对位置编码**
在一些模型中，**相同的相对位置编码**会被不同分辨率的 `query` 和 `key` 共享（比如在不同的 Transformer 块中复用同一个 `rel_pos`）。
- 如果直接共享，可能因为分辨率的不同导致不能完全匹配。
- 因此，需要进行**插值**来调整到正确的尺寸。
**(3) 插值的意义**
- 使用 **线性插值** 调整 `rel_pos`，可以**平滑地过渡**不同大小的相对位置编码。
- 这样可以保持**相对位置编码的特征一致性**，减少信息丢失或索引错位的风险。

**代码:**
1. **判断尺寸是否匹配**
    - 如果 `rel_pos` 的第一个维度（L）与 `max_rel_dist` 不一致，则需要进行**插值**。
2. **插值调整**
    - 使用 `F.interpolate` 进行**线性插值**，将 `rel_pos` 调整到 `max_rel_dist` 大小。
    - 插值前将 `rel_pos` 重塑为 `[1, L, C]`，然后转置为 `[1, C, L]`，使得插值发生在长度维度上。
    - 插值后再恢复形状为 `[L, C]`。
3. **如果已经匹配**
    - 直接使用原始 `rel_pos`。

 **为什么用线性插值？**
- **线性插值**能够平滑地估计出新位置编码值，而不会引入突兀的跳变或不连续性。
- 对于位置编码来说，平滑的过渡能够更好地保留原始编码的空间关系。

 **如果不调整，会发生什么？**
1. **索引错误**
    - 如果 `rel_pos` 长度不匹配 `q_size` 和 `k_size` 生成的 `relative_coords`，直接索引可能会越界或无法找到正确的编码。
2. **信息丢失**
    - 某些相对位置可能没有编码可用，导致注意力机制在这些位置上失效。
3. **模型性能下降**
    - 注意力机制依赖于精确的相对位置编码来捕获特征之间的空间关系。
    - 如果编码不正确，注意力分布会受到影响，最终导致性能下降。
---
**计算相对位置索引**
- **生成坐标**
    - `q_coords`：生成一个形状为 `[q_size, 1]` 的查询坐标索引。
    - `k_coords`：生成一个形状为 `[1, k_size]` 的键坐标索引。
- **坐标缩放**
    - 如果 `q_size` 和 `k_size` 不相等，则按比例缩放它们以对齐坐标。
    - 比例因子是 `max(k_size / q_size, 1.0)` 和 `max(q_size / k_size, 1.0)`。
- **计算相对坐标**
    - 通过 $q\_coords - k\_coords$ 计算查询和键之间的相对位置差。
    - 偏移 `(k_size - 1)` 保证索引始终是正数，以便在 `rel_pos` 中进行索引。
- **最终相对坐标**
    - `relative_coords` 的形状是 `[q_size, k_size]`，表示每个 `query` 和 `key` 之间的相对位置索引。
---
**提取相对位置编码**
- 使用 `relative_coords` 作为索引，从 `rel_pos_resized` 中提取相应的相对位置编码。
- `relative_coords` 的每个索引都对应一个相对位置向量（`[C]`）。
- 最终返回的张量形状为 `[q_size, k_size, C]`。

## add_decomposed_rel_pos
`add_decomposed_rel_pos` 的核心目的是**为注意力机制引入分解的相对位置编码**，从而在 Transformer 注意力机制中：
- **引入空间位置信息**：增强模型对查询 (`query`) 和键 (`key`) 之间空间相对关系的感知能力。
- **减少计算复杂度**：将二维空间相对位置编码分解为**高度方向**和**宽度方向**两个一维的相对位置编码，降低计算开销。
- **提升注意力机制的效果**：帮助模型更好地捕捉查询点与键点之间的相对空间依赖关系。
```python
def add_decomposed_rel_pos(  
    attn: torch.Tensor,  
    q: torch.Tensor,  
    rel_pos_h: torch.Tensor,  
    rel_pos_w: torch.Tensor,  
    q_size: Tuple[int, int],  
    k_size: Tuple[int, int],  
) -> torch.Tensor:  
    """  
    Calculate decomposed Relative Positional Embeddings from:paper:`mvitv2`. 
    Args:        
        attn (Tensor): attention map.        
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, 
			        C).        
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height 
					        axis.        
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width 
					        axis.        
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).    
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).  
    Returns:        
	    attn (Tensor): attention map with added relative positional 
					   embeddings.    
	    """    
    q_h, q_w = q_size  
    k_h, k_w = k_size  
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  
  
    B, _, dim = q.shape  
    r_q = q.reshape(B, q_h, q_w, dim)  
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)  
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)  
  
    attn = (  
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]  
    ).view(B, q_h * q_w, k_h * k_w)  
  
    return attn
```
1 **高度维度**和**宽度维度**获取相对位置编码:
$$R_h \in \mathbb{R}^{q_h, k_h, C}, \quad R_w \in \mathbb{R}^{q_w, k_w, C}$$
**为什么要分解成高度和宽度？**
- **减少计算复杂度**：分解成高度和宽度分别计算，避免了对整个二维空间的全连接操作，计算量大大减少。
- **提升效果**：能够分别捕获高度和宽度维度的相对位置信息。
- **更灵活**：在不同维度上可以分别进行插值和调整。

2 然后重塑Query张量.

3 **`torch.einsum`** 公式：`"bhwc,hkc->bhwk"`
- `b`：Batch size
- `h`：Query 高度
- `w`：Query 宽度
- `c`：通道维度
- `k`：Key 高度
 **运算过程**
对于每个批次 b 和每个查询点 $(q_h, q_w)$：
$$
\text{rel}_h[b, q_h, q_w, k_h] = \sum_{c=1}^{C} r_q[b, q_h, q_w, c] \cdot R_h[q_h, k_h, c]
$$
	**爱因斯坦求和约定** 可实现包括但不限于：向量内积，向量外积，矩阵乘法，转置和张量收缩

**4 将相对位置编码加到注意力图中**
- **`attn.view(B, q_h, q_w, k_h, k_w)`**
    - 将原始注意力图 `attn` 重塑为5维张量，形状为 `(B, q_h, q_w, k_h, k_w)`。
    - 这样每个查询点 `(q_h, q_w)` 都有对应的 `(k_h, k_w)` 的注意力值。
- **`rel_h[:, :, :, :, None]`**
    - 将 `rel_h` 的形状扩展为 `(B, q_h, q_w, k_h, 1)`。
    - 对应到每个查询点 `(q_h, q_w)` 的高度方向编码。
- **`rel_w[:, :, :, None, :]`**
    - 将 `rel_w` 的形状扩展为 `(B, q_h, q_w, 1, k_w)`。
    - 对应到每个查询点 `(q_h, q_w)` 的宽度方向编码。
- 将两者分别加到注意力图上。
- **`view(B, q_h * q_w, k_h * k_w)`**
    - 将注意力图还原为原始的二维形状 `(B, q_h * q_w, k_h * k_w)`。

---
 **1. 为什么要使用相对位置编码？**
 **1.1 注意力机制的局限**
- 在传统的 Transformer 注意力机制中，注意力权重由查询 (`query`) 和键 (`key`) 的特征向量通过点积计算得到：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
- 但是，这种方式**忽略了查询和键在空间上的相对位置信息**，例如：
    - 相邻像素之间的特征可能比远离像素更相关。
    - 空间方向（高度/宽度）上的关系无法被显式建模。
---
 **1.2 绝对位置编码 vs 相对位置编码**
- **绝对位置编码**：为每个位置分配一个唯一的编码。
    - 问题：当查询和键的大小不一致时，无法对齐位置编码。
- **相对位置编码**：直接建模**查询点与键点之间的相对位置关系**。
    - 更适合 Transformer 在图像任务中的场景。
---
**2. 为什么要分解成高度和宽度？**
**2.1 二维相对位置编码的计算复杂度**
- 如果直接为每对查询-键点计算二维相对位置编码，计算量是：
$$
O(q_h \times q_w \times k_h \times k_w \times d)
$$
- 当图像尺寸较大时，这种计算量非常庞大，难以接受。
**2.2 分解到高度和宽度**
- 将二维相对位置编码分解成**高度方向**和**宽度方向**两个一维编码，可以将复杂度降低到：
$$
O(q_h \times k_h \times d) + O(q_w \times k_w \times d)
$$
- 这种方法：
    - **降低计算复杂度**
    - **更易于训练**
    - **捕获更明确的方向性关系**
---
**3. `add_decomposed_rel_pos` 做了什么？**
1. **获取相对位置编码**
    - 使用 `get_rel_pos` 分别为**高度**和**宽度**维度获取相对位置编码。
2. **高度方向的相对位置编码**
    - 使用 `torch.einsum` 对查询张量和高度相对位置编码进行点积，得到高度方向的相对位置信息。
3. **宽度方向的相对位置编码**
    - 使用 `torch.einsum` 对查询张量和宽度相对位置编码进行点积，得到宽度方向的相对位置信息。
4. **将编码添加到注意力图**
    - 将高度方向和宽度方向的相对位置权重分别加到注意力图上。
---
 **4. 为什么要把相对位置编码加到注意力图中？**
在 Transformer 中，注意力图表示了查询和键之间的注意力权重。将相对位置编码加到注意力图中：
1. **显式引入位置信息**：原本的注意力分数只由特征之间的匹配度决定，加上相对位置编码后，注意力还会受到查询和键之间的空间相对位置影响。
2. **增强模型感知**：模型可以更好地理解**空间结构**和**局部关系**，例如：
    - 相邻像素之间应该有较高的注意力权重。
    - 跨越大空间距离的像素注意力权重可能较低。
3. **跨尺度泛化**：即使查询和键的尺寸不同，调整后的相对位置编码仍然可以较好地泛化到不同尺度的特征图上。

# Attention
该类实现了一个带有**多头注意力机制（Multi-Head Attention, MHA）** 和 **相对位置编码（Relative Positional Embedding, RPE）** 的注意力模块
```python
class Attention(nn.Module):  
    """Multi-head Attention block with relative position embeddings."""  
  
    def __init__(  
        self,  
        dim: int,  
        num_heads: int = 8,  
        qkv_bias: bool = True,  
        use_rel_pos: bool = False,  
        rel_pos_zero_init: bool = True,  
        input_size: Optional[Tuple[int, int]] = None,  
    ) -> None:  
        """  
        Args:            
	        dim (int): Number of input channels.            
	        num_heads (int): Number of attention heads.            
	        qkv_bias (bool):  If True, add a learnable bias to query, key, value.            
	        rel_pos (bool): If True, add relative positional embeddings to the attention map.            
	        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.            
	        input_size (tuple(int, int) or None): Input resolution for calculating the relative                
	        positional parameter size.        
        """        
        super().__init__()  
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = head_dim**-0.5  
  
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  
        self.proj = nn.Linear(dim, dim)  
  
        self.use_rel_pos = use_rel_pos  
        if self.use_rel_pos:  
            assert (  
                input_size is not None  
            ), "Input size must be provided if using relative positional encoding."  
            # initialize relative positional embeddings  
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))  
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        B, H, W, _ = x.shape  
        # qkv with shape (3, B, nHead, H * W, C)  
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  
        # q, k, v with shape (B * nHead, H * W, C)  
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)  
  
        attn = (q * self.scale) @ k.transpose(-2, -1)  
  
        if self.use_rel_pos:  
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))  
  
        attn = attn.softmax(dim=-1)  
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)  
        x = self.proj(x)  
  
        return x
```
- **`dim` (int)**：
    - 输入通道数（特征维度）。
    - 例如，如果每个像素点的特征维度是 768，那么 `dim=768`。
- **`num_heads` (int)**：
    - 注意力头的数量。
    - 每个头会独立地学习一个注意力权重。
- **`qkv_bias` (bool)**：
    - 如果为 `True`，在 Query (Q)、Key (K)、Value (V) 上添加可学习的偏置（bias）。
- **`use_rel_pos` (bool)**：
    - 是否使用**相对位置编码**。
- **`rel_pos_zero_init` (bool)**：
    - 如果为 `True`，相对位置编码参数初始化为全零。
    - 如果为 `False`，则使用随机初始化。
- **`input_size` (Optional\[Tuple\[int, int\]\])**：
    - 如果启用了相对位置编码（`use_rel_pos=True`），必须提供输入的空间分辨率 `(H, W)`。


```python 
self.num_heads = num_heads  
head_dim = dim // num_heads  
self.scale = head_dim**-0.5  
```
多头注意力机制（**Multi-Head Attention, MHA**）中的关键部分。它们定义了**注意力头的数量**、**每个头的特征维度**，以及**缩放因子**。

 **为什么要使用多头注意力？**
在原始的自注意力机制中，`Q`、`K` 和 `V` 向量使用相同的特征维度进行计算。这种单一的注意力头可能难以捕捉不同的特征子空间。
**多头注意力的优势：**
1. **信息多样性**：每个注意力头可以学习不同的注意力模式。
2. **并行计算**：多个注意力头可以并行进行点积注意力计算，提升计算效率。
3. **丰富的特征表达**：每个头可以关注不同的特征子空间，增强模型的特征表达能力。
 **维度分配**
- 假设输入特征维度为 dimdimdim（例如 768）。
- 有 num_headsnum\_headsnum_heads 个注意力头（例如 8）。
- 每个注意力头负责处理 $\text{dim} \, / \, \text{num\_heads}$ 维度的特征。
`self.scale = head_dim**-0.5` 在计算 `Q @ K^T`（点积注意力分数）时，结果的方差会随着维度 $head\_dim$ 的增加而增大。如果注意力分数的数值过大，经过 `softmax` 之后，可能会导致梯度消失或数值不稳定。
**缩放的解决方案：**
- 使用 $\sqrt{d_k}$（每个头的特征维度的平方根）来进行缩放：
$$
\frac{QK^T}{\sqrt{d_k}}
$$
- 在代码中，这等价于：
$$
\text{self.scale} = \frac{1}{\sqrt{\text{head\_dim}}}
$$

**示例：**
- 如果 `head_dim = 96`：
$$
\text{self.scale} = 96^{-0.5} \approx 0.102
$$
**效果：**
- 缩放使得注意力分数更稳定，能够更好地进行 `softmax` 操作。
- 避免注意力分布过于尖锐或过于平滑。

```python
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  
self.proj = nn.Linear(dim, dim)
```
**Transformer 自注意力机制（Self-Attention）** 中的核心部分，负责将输入特征映射到 **Query (Q)**、**Key (K)** 和 **Value (V)** 向量上，并将注意力机制的输出重新映射回原始特征空间。
在自注意力机制中：
- **Query (Q)**：表示当前特征如何与其他特征进行匹配。
- **Key (K)**：表示其他特征如何与当前特征进行匹配。
- **Value (V)**：表示特征的信息内容，将被加权并传递到下一层。

为了得到 Q, K, V，我们通常使用**线性变换**，即通过全连接层（`nn.Linear`）来学习映射关系。
Transformer 的输入是一个维度为 dim 的特征向量，我们需要： 
- 将它映射到 Q、K、V 三个不同的子空间。
- 每个子空间的维度与原始特征相同（dim）。

 **Q 和 K**：计算注意力分数（Attention Score）。
 **V**：在注意力分数的加权下进行信息聚合。

**含义**
- **输入**：特征维度为 `dim` 的张量。
- **输出**：将输入特征映射到 `dim * 3` 的维度。
    - 其中，`dim` 的 1/3 用于 `Q`（查询向量）。
    - 1/3 用于 `K`（键向量）。
    - 1/3 用于 `V`（值向量）。
- **bias=qkv_bias**：为每个映射添加可学习的偏置项。
**数学表达**
假设输入是一个张量 $X \in \mathbb{R}^{B \times N \times \text{dim}}$，其中：
- B：批量大小
- N：序列长度（例如图像的像素点数量或序列的 token 数）
- dim：特征维度
$$
[Q, K, V] = X W_{\text{qkv}} + b_{\text{qkv}}
$$
$W_{\text{qkv}}$: 权重矩阵，形状为$\mathbb{R}^{\text{dim} \times (3 \times \text{dim})}$
$b_{\text{qkv}}:$  可选偏置项
最终，输出的张量形状为：$$(B,N,3×dim)$$
然后，可以使用 `torch.chunk` 或 `torch.unbind` 将其分割成：
$$
Q \in \mathbb{R}^{B \times N \times \text{dim}}
$$
$$
K \in \mathbb{R}^{B \times N \times \text{dim}}
$$
$$
V \in \mathbb{R}^{B \times N \times \text{dim}}
$$
```python
self.proj = nn.Linear(dim, dim)
```
- 将注意力机制输出的特征再次映射回原始维度 `dim`。
- 这样做的目的是：
    1. 保持特征维度一致，方便与后续的残差连接（Residual Connection）进行加法操作。
    2. 提供额外的学习能力，允许模型调整注意力后的特征表示。
假设注意力机制输出的张量是$Z \in \mathbb{R}^{B \times N \times \text{dim}}$
$$
\text{Output} = Z W_{\text{proj}} + b_{\text{proj}}
$$
$W_{\text{proj}}$: 权重矩阵，形状为$\mathbb{R}^{\text{dim} \times \text{dim}}$
$b_{\text{proj}}:$  可选偏置项

---

**前向传播过程**
获取 Query (Q), Key (K), Value (V)
```python
qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
```
1. **输入**：`x` 的形状为 `(B, H, W, dim)`
2. **线性映射**：`qkv` 的形状为 `(B, H * W, 3 * dim)`
3. **重塑**：将 `qkv` 分为 Query、Key、Value：
    - 形状调整为 `(3, B, num_heads, H * W, head_dim)`
4. **分离 Q, K, V**：使用 `unbind` 将 `qkv` 分成 Q、K、V：
    - 每个形状为 `(B * num_heads, H * W, head_dim)`
```python
attn = (q * self.scale) @ k.transpose(-2, -1)
```
- 对每个注意力头：
    
    - `q` 和 `k` 进行**缩放点积注意力**。
    - 这里的缩放因子是 $\text{self.scale} = \frac{1}{\sqrt{\text{head\_dim}}}$ ，用来防止注意力分数过大导致梯度消失。
- 注意力分数形状为 `(B * num_heads, H * W, H * W)`
- 添加相对位置编码
```python
if self.use_rel_pos:
    attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
```
- 如果 `use_rel_pos` 为 `True`，使用 `add_decomposed_rel_pos` 将**高度**和**宽度**的相对位置编码加到注意力分数上。
	- **目的**：引入查询 (`q`) 和键 (`k`) 在空间上高度和宽度的相对位置信息。
	- 调整后的 `attn` 保留了位置关系。
- 计算注意力加权的 Value (V)
```python
attn = attn.softmax(dim=-1)
x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
```
1. **Softmax**：对注意力分数进行 `softmax` 归一化。
2. **加权 Value**：将注意力权重 `attn` 作用在 `v` 上。
3. **重塑回原形状**：将结果重塑回 `(B, H, W, dim)`。
- 线性映射回原空间

# window_partition

该函数的核心目的是**将输入的特征图分割成多个不重叠的窗口**（Windows），在必要时进行**填充（Padding）**，以确保每个窗口的尺寸一致。这种操作常见于 **视觉Transformer（如 Swin Transformer）** 中，用于处理图像或特征图的局部区域信息。
通过这样可以实现: 
1. **局部感知**
    - 将图像或特征图分割成小窗口，模型可以更有效地捕捉局部细节。
2. **并行计算**
    - 小窗口之间可以并行进行注意力机制计算，提升计算效率。
3. **填充确保完整性**
    - 保证分割后的每个窗口都是完整的，避免信息丢失。
```python
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:  
    """  
    Partition into non-overlapping windows with padding if needed.    
    Args:        
	    x (tensor): input tokens with [B, H, W, C].        
	    window_size (int): window size.  
    Returns:        
	    windows: windows after partition with [B * num_windows, window_size, window_size, C].        
	    (Hp, Wp): padded height and width before partition    
	    """    
	B, H, W, C = x.shape  
  
    pad_h = (window_size - H % window_size) % window_size  
    pad_w = (window_size - W % window_size) % window_size  
    if pad_h > 0 or pad_w > 0:  
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  
    Hp, Wp = H + pad_h, W + pad_w  
  
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)  
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  
    return windows, (Hp, Wp)
```
- **`H % window_size`**：高度不能被窗口大小整除时，会有剩余部分。
- **`window_size - H % window_size`**：计算需要填充多少像素，才能使高度整除窗口大小。
- **`% window_size`**：如果高度本身已经可以整除窗口大小，则填充为 `0`。
- **`F.pad`** 对张量进行填充，填充顺序为：
    - `(左, 右, 上, 下)`：`(0, 0, 0, pad_w, 0, pad_h)`
    - `左、右` 和 `上、下` 的维度是 `C` 和 `W`，分别进行零填充。
- **效果**：
    - 特征图高度变为 `H + pad_h`
    - 特征图宽度变为 `W + pad_w`
1. **`Hp // window_size` 和 `Wp // window_size`**
    - 将高度和宽度划分成 `Hp / window_size` 和 `Wp / window_size` 个小窗口。
2. **`x.view`**
    - 将特征图重塑为：
$$
(B, H_p \, / \, \text{window\_size}, \text{window\_size}, W_p \, / \, \text{window\_size}, \text{window\_size}, C)
$$
- 每个维度的含义：
    - `B`：批量大小
    - `Hp // window_size`：窗口在高度方向上的个数
    - `window_size`：每个窗口的高度
    - `Wp // window_size`：窗口在宽度方向上的个数
    - `window_size`：每个窗口的宽度
    - `C`：通道数

**重新排列窗口**
1. **`permute`**
    - 将维度重新排列：
$$
(B, H_p \, / \, \text{window\_size}, \text{window\_size}, W_p \, / \, \text{window\_size}, \text{window\_size}, C)
$$
2. **`contiguous`**
    - 保证内存布局是连续的，避免出现错误。
3. **`view`**
    - 将所有窗口压缩到一个批次中：
$$
(B \times \text{num\_windows}, \text{window\_size}, \text{window\_size}, C)
$$
- `num_windows = (Hp // window_size) * (Wp // window_size)`

# window_unpartition
**目的**：  
将通过 `window_partition` 分割并填充的窗口重新拼接回**原始特征图**，并移除多余的填充。
**场景**：  
在 **视觉Transformer** 中，特征图被分成多个窗口进行局部注意力计算。完成局部计算后，这些窗口需要**还原**到原始的特征图中，并去除在 `window_partition` 阶段添加的多余填充。
```python 
def window_unpartition(  
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]  
) -> torch.Tensor:  
    """  
    Window unpartition into original sequences and removing padding.    
    Args:        
	    windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].        
	    window_size (int): window size.        
	    pad_hw (Tuple): padded height and width (Hp, Wp).        
	    hw (Tuple): original height and width (H, W) before padding.  
    Returns:        
	    x: unpartitioned sequences with [B, H, W, C].    
	"""    
	Hp, Wp = pad_hw  
    H, W = hw  
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)  
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)  
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)  
  
    if Hp > H or Wp > W:  
        x = x[:, :H, :W, :].contiguous()  
    return x
```

- 计算批量大小：
$$
\frac{B \times \text{num\_windows}}{\frac{H_p}{\text{window\_size}} \times \frac{W_p}{\text{window\_size}}}
$$
- 重塑窗口张量
$$
(B, H_p \, / \, \text{window\_size}, W_p \, / \, \text{window\_size}, \text{window\_size}, \text{window\_size}, C)
$$
- 调整维度顺序
- 去除填充
	- 如果填充后的尺寸 `Hp` 和 `Wp` 大于原始尺寸 `H` 和 `W`，则裁剪多余的部分。
	- **`x[:, :H, :W, :]`**：仅保留原始特征图的高度 `H` 和宽度 `W`。
	- **`contiguous`**：确保裁剪后的内存布局连续。
 
 **为什么不是四个维度？**:
**四个维度不足以表达所有信息**：
- **如果直接用四个维度（B, Hp, Wp, C）**：
    - 无法保留窗口内部的空间信息。
    - `window_size × window_size` 的局部块信息将被打散。
**五个维度的优势**：
1. **空间信息保持**
    - `(Hp // window_size, Wp // window_size)` 负责定位每个窗口在原始特征图中的位置。
    - `(window_size, window_size)` 负责保留窗口内部的空间排列。
2. **还原性**
    - 这五个维度确保了可以轻松还原到原始特征图的空间结构。
    - 后续使用 `permute` 和 `view` 操作将这些窗口拼接回 `(B, Hp, Wp, C)`。

# Block
该类是一个 **Transformer 块**，支持：
- **窗口注意力机制（Window Attention）**
- **全局注意力机制（Global Attention）**（当 `window_size = 0` 时）
- **残差连接（Residual Connection）**
- **MLP 块（MLP Block）**
这种结构常见于视觉 Transformer（如 **Swin Transformer**），用于高效地捕捉图像特征的局部和全局依赖关系。
```python
class Block(nn.Module):  
    """Transformer blocks with support of window attention and residual propagation blocks"""  
  
    def __init__(  
        self,  
        dim: int,  
        num_heads: int,  
        mlp_ratio: float = 4.0,  
        qkv_bias: bool = True,  
        norm_layer: Type[nn.Module] = nn.LayerNorm,  
        act_layer: Type[nn.Module] = nn.GELU,  
        use_rel_pos: bool = False,  
        rel_pos_zero_init: bool = True,  
        window_size: int = 0,  
        input_size: Optional[Tuple[int, int]] = None,  
    ) -> None:  
        """  
        Args:            
	        dim (int): Number of input channels.            
	        num_heads (int): Number of attention heads in each ViT block.   
	        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.    
	        qkv_bias (bool): If True, add a learnable bias to query, key, value.            
	        norm_layer (nn.Module): Normalization layer.            act_layer (nn.Module): Activation layer.            
	        use_rel_pos (bool): If True, add relative positional embeddings to the attention map.            
	        rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.            
	        window_size (int): Window size for window attention blocks. If it equals 0, then use global attention.            
	        input_size (tuple(int, int) or None): Input resolution for calculating the relative positional parameter size.        
	    """        
	    super().__init__()  
        self.norm1 = norm_layer(dim)  
        self.attn = Attention(  
            dim,  
            num_heads=num_heads,  
            qkv_bias=qkv_bias,  
            use_rel_pos=use_rel_pos,  
            rel_pos_zero_init=rel_pos_zero_init,  
            input_size=input_size if window_size == 0 else (window_size, window_size),  
        )  
  
        self.norm2 = norm_layer(dim)  
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)  
  
        self.window_size = window_size  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        shortcut = x  
        x = self.norm1(x)  
        # Window partition  
        if self.window_size > 0:  
            H, W = x.shape[1], x.shape[2]  
            x, pad_hw = window_partition(x, self.window_size)  
  
        x = self.attn(x)  
        # Reverse window partition  
        if self.window_size > 0:  
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))  
  
        x = shortcut + x  
        x = x + self.mlp(self.norm2(x))  
  
        return x
```