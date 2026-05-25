import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Flickr8kCaptionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        processor,
        tokenizer,
        max_images=200,
        max_length=32,
        use_all_captions=True,
        prompt="A photo of",
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "Images")
        self.caption_file = os.path.join(root_dir, "captions.txt")

        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt

        df = pd.read_csv(self.caption_file)

        image_names = df["image"].drop_duplicates().tolist()[:max_images]

        samples = []

        for image_name in image_names:
            rows = df[df["image"] == image_name]

            if use_all_captions:
                for _, row in rows.iterrows():
                    samples.append({
                        "image": image_name,
                        "caption": row["caption"]
                    })
            else:
                row = rows.iloc[0]
                samples.append({
                    "image": image_name,
                    "caption": row["caption"]
                })

        self.samples = samples

        print(f"Loaded {len(self.samples)} caption samples from {len(image_names)} images.")

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text, max_length, add_eos=False):
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1 if add_eos else max_length,
        )

        input_ids = encoded["input_ids"]

        if add_eos:
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        attention_mask = [1] * len(input_ids)

        pad_length = max_length - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")

        caption = item["caption"].strip()

        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # prompt 单独编码，训练时作为条件输入，但不计算 prompt loss
        prompt_input_ids, prompt_attention_mask = self.encode_text(
            self.prompt,
            max_length=8,
            add_eos=False,
        )

        # caption 单独编码，训练时只对 caption 部分计算 loss
        input_ids, attention_mask = self.encode_text(
            caption,
            max_length=self.max_length,
            add_eos=True,
        )

        return {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "image_path": image_path,
        }