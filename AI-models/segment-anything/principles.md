
# 本文贡献
segment anything 模型由facebook meta AI团队开发，通过三个互相关联的组件构建了一个基础模型，这三个基础组件分别是：
- a prompt-able segmentation task（可提示的分割任务）
- a segmentation model (SAM) that powers data annotation and enables zero-shot transfer to a range of tasks via prompt engineering （一个驱动数据注释并通过提示工程实现零样本转移到各种任务的分割模型（SAM））
- a data engine for collecting SA-1B, our dataset of over 1 billion masks （一个用于收集SA-1B（超过10亿个掩码数据集）的数据引擎。）
# 摘要
+ 分割数据集：数据集包含1100万张图像（11M ），超过10亿个mask掩码
+ prompt-able model: 可zero shot 迁移至新的distribution和task -> 零样本性能优于prior fully supervisied results
# Introduction
解决以下三个问题：
1. 什么**任务**将实现零样本推理？
2. 对应的**模型**架构是什么？
3. 什么**数据**可以支持这个任务和模型？

## 1. 任务：
promptable task -> is to return a valid segmentation mask given any segmentation prompt 
## 2. 模型：
一个强大的图像编码器计算图像嵌入，一个提示编码器嵌入提示，然后这两种信息源结合在一个轻量级掩码解码器中，预测分割掩码。通过将SAM分为图像编码器和快速提示编码器/掩码解码器，同一图像嵌入可以与不同的提示一起重新使用（并摊销其成本）。给定图像嵌入，提示编码器和掩码解码器可以在网页浏览器中以约50ms从提示预测出掩码。我们专注于点、框和掩码提示，也介绍了自由形式文本提示的初步结果。
```
The promptable segmentation task and the goal
of real-world use impose constraints on the model architec-
ture. In particular, the model must support flexible prompts,
needs to compute masks in amortized real-time to allow in-
teractive use, and must be ambiguity-aware. Surprisingly,
we find that a simple design satisfies all three constraints:
a powerful image encoder computes an image embedding,
a prompt encoder embeds prompts, and then the two infor-
mation sources are combined in a lightweight mask decoder
that predicts segmentation masks. We refer to this model as
the Segment Anything Model, or SAM (see Fig. 1b). By
separating SAM into an image encoder and a fast prompt
encoder / mask decoder, the same image embedding can
be reused (and its cost amortized) with different prompts.
Given an image embedding, the prompt encoder and mask
decoder predict a mask from a prompt in ∼50ms in a web
browser. We focus on point, box, and mask prompts, and
also present initial results with free-form text prompts. To
make SAM ambiguity-aware, we design it to predict mul-
tiple masks for a single prompt allowing SAM to naturally
handle ambiguity, such as the shirt vs. person example
```
> 可提示的分割任务和实际使用中的目标对模型架构提出了约束。特别是，模型必须支持灵活的提示，需要以摊销后的实时计算掩码以允许交互使用，并且必须对歧义具有感知能力。令人惊讶的是，我们发现一个简单的设计可以满足所有这三项约束：一个强大的图像编码器计算图像嵌入，一个提示编码器嵌入提示，然后这两种信息源结合在一个轻量级掩码解码器中，预测分割掩码。我们将这种模型称为Segment Anything Model，或简称SAM（见图1b）。通过将SAM分为图像编码器和快速提示编码器/掩码解码器，同一图像嵌入可以与不同的提示一起重新使用（并摊销其成本）。给定图像嵌入，提示编码器和掩码解码器可以在网页浏览器中以约50ms从提示预测出掩码。我们专注于点、框和掩码提示，也介绍了自由形式文本提示的初步结果。为了使SAM具有歧义感知，我们设计它以便为单个提示预测多个掩码，使SAM自然处理歧义，例如衬衫与人的示例。


# Segment Anything Task

prompt可以是一组前景/背景点、一个粗糙的框或掩码、自由形式的文本（指示在图像中分割什么的任何信息），根据提示返回一个有效的分割掩码。“有效”意味着，即使提示模棱两可，可以引用多个对象（例如衬衫与人的例子），输出也至少是其中一个合理的掩码。类似于期望一个语言模型对一个模糊的提示输出一个一致的响应。

1. **可提示分割任务**：这段内容介绍了一种被称为“可提示分割任务”（promptable segmentation task）的预训练算法。在这个任务中，模型会针对每个训练样本模拟一系列的提示（如点、框、掩码），然后将模型预测的掩码与真实标签（ground truth）进行比较。
    
2. **方法的来源**：这种方法借鉴了交互式分割（interactive segmentation）的方法[109, 70]。在交互式分割中，模型的目标是通过逐步接收用户输入，最终预测出一个有效的掩码。
    
3. **任务目标的不同**：与交互式分割不同，这里的目标是无论提示是否明确，模型都能即时预测出一个有效的掩码。这意味着即使提示是模糊的（例如一个点可能对应多个对象），模型也必须能够输出一个合理的掩码。这种设定确保了预训练模型在处理涉及模糊提示的应用场景（如自动注释）时也能表现良好。
    
4. **挑战和需求**：要在这个任务上表现良好是具有挑战性的，因为这需要专门的建模和训练损失选择。文章在第3节中会进一步讨论这些专门的建模和训练策略。

根据文章内容，作者的主要工作组件是**可提示分割模型**（promptable segmentation model）。以下是其主要工作方式的详细解释：

1. **任务泛化能力**：可提示分割模型具备一种称为任务泛化（task generalization）的能力。这意味着该模型可以在推理（inference）时执行与训练任务不同的新任务。这与多任务分割系统（multi-task segmentation systems）不同。多任务分割系统中的单一模型执行一组固定的任务（如联合语义分割、实例分割和全景分割），这些任务在训练和测试阶段都是一致的。

2. **模块化设计**：可提示分割模型被设计为一个灵活的组件，可以在不同的系统中用于不同的任务。例如，为了执行实例分割任务，模型可以与现有的对象检测器结合使用。在实例分割任务中，首先使用对象检测器检测图像中的对象并生成边界框，然后将这些边界框作为提示输入到可提示分割模型中，模型会根据这些提示生成相应的实例分割掩码。

3. **具体工作方式**：
   - **训练阶段**：在训练阶段，可提示分割模型会模拟一系列提示（如点、框、掩码）并将模型的掩码预测与真实标签进行比较。该方法借鉴了交互式分割，但目标是无论提示是否明确，模型都能即时预测出一个有效的掩码。
   - **推理阶段**：在推理阶段，模型可以通过与其他组件（如对象检测器）的结合来执行新的任务。例如，在实例分割任务中，模型会根据对象检测器生成的边界框提示来生成分割掩码。

总结来说，作者的主要工作组件是**可提示分割模型**，它通过处理各种提示（如点、框、掩码）来生成分割掩码。与多任务分割系统不同，这种模型能够在推理时执行与训练任务不同的新任务，通过与其他组件（如对象检测器）的结合来实现复杂的分割任务。

# Segment Anything Model
![[Segment Anything Model overview.png]]

如上图所示，Segment Anything 有三个部分组成：an image encoder, a flexible prompt encoder, and a fast mask decoder.

模型基于Transform Vision, 并对实时性能做了权衡

**Image encoder.**  , 受可扩展性和强大的预训练方法的启发，我们使用了 MAE 预训练的 Vision Transformer (ViT)，该模型经过最低限度的调整，可以处理高分辨率输入。图像编码器每幅图像运行一次，可以在提示模型之前应用。

**Prompt encoder.**  我们考虑两组提示：稀疏（点、框、文本）和密集（掩码）。我们用位置编码表示点和框，并用 CLIP 提供的现成文本编码器对每种提示类型和自由格式文本的学习嵌入进行求和。密集提示（即掩码）使用卷积进行嵌入，并与图像嵌入逐元素求和。

**Mask decoder**. 掩码解码器有效地将图像嵌入、提示嵌入和输出标记映射到掩码。此设计受到"Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with Transformers" 和 “Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation” 的启发，采用了 Transformer 解码器块 的修改，后跟动态掩码预测头。我们修改后的解码器块在两个方向（提示到图像嵌入和反之亦然）使用提示自注意和交叉注意来更新所有嵌入。运行两个块后，我们对图像嵌入进行上采样，MLP 将输出标记映射到动态线性分类器，然后计算每个图像位置的掩码前景概率

**Resolving ambiguity**. 如果给出一个模糊提示，模型将使用一个输出对多个有效掩码进行平均。
为了解决这个问题，我们修改了模型，以预测单个提示的多个输出掩码（见图 3）。我们发现 3 个掩码输出足以解决大多数常见情况（嵌套掩码通常最多为三层：整体、部分和子部分）。在训练期间，我们仅反向传播掩码上的最小损失。为了对掩码进行排序，模型会预测每个掩码的置信度分数（即估计的 IoU）。

效率。整体模型设计主要受效率驱动。给定预先计算的图像嵌入，提示编码器和掩码解码器在 CPU 上的网络浏览器中运行，大约需要 50 毫秒。这种运行时性能使我们的模型能够无缝、实时地进行交互式提示。

损失和训练。我们使用 [14] 中使用的焦点损失 [65] 和骰子损失 [73] 的线性组合来监督掩码预测。我们使用混合的几何提示来训练可提示的分割任务（有关文本提示，请参阅 §7.5）。按照 [92, 37]，我们通过对每个掩码随机抽样 11 轮提示来模拟交互式设置，从而使 SAM 能够无缝集成到我们的数据引擎中。





# Segment Anything Data Engine