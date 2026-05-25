# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片

## 4. 模型结构

请说明自己的 Mini-BLIP2 结构

```text
本实验实现了一个简化版 Mini-BLIP2 模型，用于完成图像描述生成任务。整体结构如下：
Image → Frozen Vision Encoder → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption
具体流程为：首先将输入图片送入冻结的视觉编码器 CLIP Vision Encoder，提取图像 patch 特征；然后使用可训练的 Mini Q-Former，通过 learnable query tokens 对图像特征进行 cross-attention，得到与文本生成相关的视觉表示；接着使用 Projection Layer 将 Q-Former 输出映射到语言模型 OPT-125M 的 hidden size；最后将映射后的视觉 prefix 与文本输入拼接，送入冻结的 OPT Language Decoder 中进行 caption 生成。
在训练过程中，CLIP Vision Encoder 和 OPT Language Decoder 的参数均被冻结，只训练 Mini Q-Former 和 Projection Layer。
### 4.1 Vision Encoder
```

### 4.1 Vision Encoder

填写使用的视觉编码器，openai/clip-vit-base-patch32
该模块使用 Hugging Face Transformers 中的 `CLIPVisionModel` 加载。输入图片经过 CLIP 图像预处理后，送入 CLIP Vision Encoder，输出图像特征序列。该视觉编码器在训练过程中保持冻结，不参与参数更新。
- 模型名称：openai/clip-vit-base-patch32
- 加载方式：CLIPVisionModel
- 是否冻结：是
- 输出 hidden size：768
### 4.2 Mini Q-Former

说明自己实现的 Mini Q-Former：16

- query token 数量：768
- hidden size：2
- Transformer 层数：8
- 是否使用 cross-attention：是
具体来说，CLIP Vision Encoder 输出的图像特征作为 memory，可学习 query tokens 作为 target，输入到 `nn.TransformerDecoder` 中。通过 cross-attention，query tokens 能够从图像 patch 特征中聚合视觉信息。最终 Mini Q-Former 输出形状为：
[batch_size, num_query_tokens, hidden_size]
即：
[batch_size, 16, 768]
### 4.3 Language Decoder

填写使用的语言解码器，本实验使用的语言解码器为：
facebook/opt-125m
该模块使用 Hugging Face Transformers 中的 `OPTForCausalLM` 加载。OPT-125M 作为 frozen language decoder，用于根据视觉 prefix 和文本上下文生成英文 caption。
- 模型名称：facebook/opt-125m
- 加载方式：OPTForCausalLM
- 是否冻结：是
- 词表大小：50272
- hidden size：768
训练时，OPT-125M 的参数不更新，但梯度仍然需要从 loss 反向传播经过 OPT 的输入 embedding 回到 Projection Layer 和 Mini Q-Former。因此在 forward 过程中没有对 OPT 使用 `torch.no_grad()`，只通过 `requires_grad=False` 冻结其参数。
## 5. 训练设置

请填写：

- 训练数据量：
- epoch：3
- batch size：1
- learning rate：1e-4
- optimizer：AdamW 
- loss function：Cross Entropy Loss 
- 冻结的模块：CLIP Vision Encoder；OPT-125M Language Decoder
- 训练的模块：Mini Q-Former；Projection Layer
训练过程中，视觉编码器 `openai/clip-vit-base-patch32` 和语言解码器 `facebook/opt-125m` 均保持冻结，不参与参数更新。模型主要训练中间的 Mini Q-Former 和 Projection Layer，使其能够将图像特征映射到语言模型可以利用的 embedding 空间中，从而完成图像描述生成任务。
## 6. 训练过程

粘贴训练日志或 loss 变化截图。

示例：

| Epoch | Train Loss |
|---|---:|
| 1 |  |
| 2 |  |
| 3 |  |

## 7. 生成结果展示

至少展示 3—5 个例子。

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

如果方便，可以把图片也插入报告中。

## 8. 总结

请简要说明：

- 是否成功跑通训练；
- 生成效果如何；
- 遇到了什么问题；
- 如果继续改进，可以怎么做。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：例如 entir.io
- 对话链接：
- 使用的 AI 模型：例如 Claude / ChatGPT / Gemini
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
（在这里写）
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出（或截图）：

```text
（在这里粘贴 git log --oneline）
```
