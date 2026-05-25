import torch
import torch.nn as nn
from transformers import CLIPVisionModel, OPTForCausalLM


class MiniQFormer(nn.Module):
    """
    简化版 Q-Former：
    - 使用一组可学习 query tokens
    - 通过 TransformerDecoderLayer 对图像特征做 cross-attention
    """

    def __init__(
        self,
        vision_hidden_size=768,
        num_query_tokens=16,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, vision_hidden_size)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vision_hidden_size,
            nhead=num_heads,
            dim_feedforward=vision_hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(vision_hidden_size)

    def forward(self, image_embeds):
        """
        image_embeds: [batch_size, num_patches, vision_hidden_size]
        return: [batch_size, num_query_tokens, vision_hidden_size]
        """
        batch_size = image_embeds.size(0)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        query_output = self.decoder(
            tgt=query_tokens,
            memory=image_embeds,
        )

        query_output = self.norm(query_output)

        return query_output


class MiniBLIP2(nn.Module):
    """
    Mini-BLIP2:
    Frozen CLIP Vision Encoder
        ↓
    Trainable Mini Q-Former
        ↓
    Trainable Projection Layer
        ↓
    Frozen OPT Language Decoder
    """

    def __init__(
        self,
        vision_model_path,
        language_model_path,
        num_query_tokens=16,
        qformer_layers=2,
        qformer_heads=8,
    ):
        super().__init__()

        # 1. Frozen Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_path,
            local_files_only=True,
        )

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        vision_hidden_size = self.vision_encoder.config.hidden_size

        # 2. Trainable Mini Q-Former
        self.qformer = MiniQFormer(
            vision_hidden_size=vision_hidden_size,
            num_query_tokens=num_query_tokens,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
        )

        # 3. Frozen Language Decoder
        self.language_model = OPTForCausalLM.from_pretrained(
            language_model_path,
            local_files_only=True,
        )

        for param in self.language_model.parameters():
            param.requires_grad = False

        language_hidden_size = self.language_model.get_input_embeddings().embedding_dim

        # 4. Trainable Projection Layer
        self.projection = nn.Linear(
            vision_hidden_size,
            language_hidden_size,
        )

        self.num_query_tokens = num_query_tokens

    def forward(
            self,
            pixel_values,
            input_ids,
            attention_mask,
            prompt_input_ids=None,
            prompt_attention_mask=None,
    ):
        """
        pixel_values: [batch_size, 3, 224, 224]
        input_ids: caption token ids, [batch_size, seq_len]
        attention_mask: caption attention mask, [batch_size, seq_len]
        prompt_input_ids: prompt token ids, [batch_size, prompt_len]
        prompt_attention_mask: prompt attention mask, [batch_size, prompt_len]
        """

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. Frozen CLIP Vision Encoder
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state

        # 2. Mini Q-Former
        query_output = self.qformer(image_embeds)

        # 3. Projection Layer
        prefix_embeds = self.projection(query_output)

        prefix_attention_mask = torch.ones(
            batch_size,
            self.num_query_tokens,
            dtype=attention_mask.dtype,
            device=device,
        )

        # 4. Caption embedding
        caption_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 5. 如果提供了 prompt，则拼接 prompt
        if prompt_input_ids is not None and prompt_attention_mask is not None:
            prompt_embeds = self.language_model.get_input_embeddings()(prompt_input_ids)

            inputs_embeds = torch.cat(
                [prefix_embeds, prompt_embeds, caption_embeds],
                dim=1,
            )

            full_attention_mask = torch.cat(
                [prefix_attention_mask, prompt_attention_mask, attention_mask],
                dim=1,
            )

            prefix_labels = torch.full(
                (batch_size, self.num_query_tokens),
                -100,
                dtype=input_ids.dtype,
                device=device,
            )

            prompt_labels = torch.full(
                prompt_input_ids.shape,
                -100,
                dtype=input_ids.dtype,
                device=device,
            )

            caption_labels = input_ids.clone()
            caption_labels[attention_mask == 0] = -100

            full_labels = torch.cat(
                [prefix_labels, prompt_labels, caption_labels],
                dim=1,
            )

        else:
            inputs_embeds = torch.cat(
                [prefix_embeds, caption_embeds],
                dim=1,
            )

            full_attention_mask = torch.cat(
                [prefix_attention_mask, attention_mask],
                dim=1,
            )

            prefix_labels = torch.full(
                (batch_size, self.num_query_tokens),
                -100,
                dtype=input_ids.dtype,
                device=device,
            )

            caption_labels = input_ids.clone()
            caption_labels[attention_mask == 0] = -100

            full_labels = torch.cat(
                [prefix_labels, caption_labels],
                dim=1,
            )

        # 不能 torch.no_grad()，否则梯度无法回传到 qformer 和 projection
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

        return outputs
    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values,
        tokenizer,
        max_new_tokens=16,
        min_new_tokens=3,
        num_beams=1,
        prompt="A photo of",
    ):
        """
        根据输入图片生成英文 caption。

        pixel_values: [batch_size, 3, 224, 224]
        tokenizer: OPT tokenizer
        """

        self.eval()

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. Frozen CLIP vision encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        # 2. Mini Q-Former
        query_output = self.qformer(image_embeds)

        # 3. Projection Layer
        prefix_embeds = self.projection(query_output)

        # 4. prefix attention mask
        prefix_attention_mask = torch.ones(
            batch_size,
            self.num_query_tokens,
            dtype=torch.long,
            device=device,
        )

        # 5. 加入文本 prompt，避免 OPT 直接输出 eos
        prompt_inputs = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        prompt_input_ids = prompt_inputs["input_ids"].to(device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

        prompt_embeds = self.language_model.get_input_embeddings()(prompt_input_ids)

        # 6. 拼接图像 prefix 和文本 prompt
        inputs_embeds = torch.cat(
            [prefix_embeds, prompt_embeds],
            dim=1,
        )

        attention_mask = torch.cat(
            [prefix_attention_mask, prompt_attention_mask],
            dim=1,
        )

        # 7. 使用 OPT 生成 caption
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=1,
            do_sample=False,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        captions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        cleaned_captions = []

        for cap in captions:
            cap = cap.strip()

            # 去掉换行
            cap = cap.replace("\n", " ").strip()

            # 如果生成了多个句子，只保留第一句
            if "." in cap:
                cap = cap.split(".")[0].strip() + "."

            # 兼容不同 decode 行为
            if cap == "":
                cap = prompt + "."
            elif cap.lower().startswith(prompt.lower()):
                cap = cap.strip()
            else:
                cap = prompt + " " + cap

            # 清理空格和重复句号
            cap = " ".join(cap.split())
            cap = cap.replace(" .", ".")
            cap = cap.replace(". .", ".")
            cap = cap.replace("..", ".")

            cleaned_captions.append(cap.strip())

        return cleaned_captions
    def print_trainable_parameters(self):
        total_params = 0
        trainable_params = 0

        for _, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")