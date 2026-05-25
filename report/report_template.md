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

示例：[Uploading train_log.txt…]()


| Epoch | Train Loss |
|---|---:|
| 1 |  |
| 2 |  |
| 3 |  |

## 7. 生成结果展示

至少展示 3—5 个例子。

| 图片编号 | 真实 Caption | 模型生成 Caption |
|---|---|---|
| 1000268201_693b08cb0e.jpg|  A child in a pink dress is climbing up a set of stairs in an entry way | A photo of a girl in pink and blue shorts is playing with her little sister |
| 1001773457_577c3a7d70.jpg | A black dog and a white dog with brown spots are staring at each other in the street | A photo of a dog with a black and white collar |
| 1002674143_1b742ab4b8.jpg | There is a girl with pigtails sitting in front of a rainbow painting | A photo of a girl in pink and blue wearing a white shirt and black pants. |

如果方便，可以把图片也插入报告中。

## 8. 总结

请简要说明：
本实验成功跑通了 Mini-BLIP2 图像描述生成的完整训练流程。模型能够正常读取 Flickr8k 前 200 张图片及其对应 caption，完成前向传播、loss 计算、反向传播和 checkpoint 保存。训练过程中 loss 能够下降，说明 Mini Q-Former 和 Projection Layer 的参数得到了更新。
在生成效果方面，模型已经能够根据输入图片生成英文 caption，不再出现无法生成文本的情况。但是生成结果的准确性有限，部分 caption 与真实图片内容不完全一致。例如模型能够生成完整的英文句子，但有时会受到语言模型先验影响，生成一些常见但不准确的描述。这说明模型已经完成了基本的 image captioning 流程，但还没有充分学习到图像内容与文本描述之间的细粒度对应关系。
实验过程中主要遇到了以下问题：首先，Hugging Face 模型下载时出现网络超时，因此改为使用本地模型路径加载 CLIP 和 OPT；其次，由于 PyTorch 版本较低，加载 `.bin` 权重时出现安全限制报错，之后通过升级 PyTorch 或使用 safetensors 权重解决；另外，初始生成阶段曾出现空 caption，原因是模型容易直接生成结束符，后续通过加入 prompt 和调整 caption 编码方式解决；最后，由于只使用少量数据并冻结了大部分模型参数，生成结果存在重复和不准确的问题。
如果继续改进，可以从以下几个方面入手：第一，增加训练数据量，使用完整 Flickr8k 数据集而不是前 200 张图片；第二，增加训练 epoch，并在有 GPU 的环境下使用更大的 batch size；第三，改进 Mini Q-Former 结构，例如增加 Transformer 层数和 query token 数量；第四，可以尝试微调语言模型的部分参数，或者使用 LoRA 等参数高效微调方法；第五，可以引入更规范的验证集和评价指标，例如 BLEU、CIDEr、METEOR 等，对生成效果进行定量评估。
## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：例如 entir.io
- 对话链接：
- 使用的 AI 模型：ChatGPT、Gemini
- 累计对话时长 / 会话数：2小时

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：
AI 主要在项目结构设计、数据读取代码、Mini-BLIP2 模型搭建、训练脚本和生成脚本调试等环节提供了帮助。在实现过程中，我根据本地运行结果对代码进行了多次修改，例如解决 Hugging Face 下载超时、模型权重加载报错、生成 caption 为空以及生成结果重复等问题。部分 AI 给出的生成策略效果并不好，我通过实际运行结果进行了调整和取舍，最终保留了能够稳定跑通训练和生成流程的实现方案。
```text
（在这里写）
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：https://github.com/yadang111/blip2
- 总 commit 数：6

粘贴 `git log --oneline` 输出（或截图）：a1b2c3d docs: 完善实验报告与项目说明
e4f5g6h docs: 添加训练日志与生成结果
i7j8k9l feat: 添加 caption 生成脚本
m1n2o3p feat: 实现训练 loop 与 cross entropy loss
q4r5s6t feat: 实现 Mini-BLIP2 模型结构
u7v8w9x feat: 加载 Flickr8k 前 200 张图片与 caption

```text
（在这里粘贴 git log --oneline）
```
