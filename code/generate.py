import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, AutoTokenizer

from dataset import Flickr8kCaptionDataset
from model import MiniBLIP2


def generate():
    # =========================
    # 1. 路径配置
    # =========================
    root_dir = r"E:\doc\lhf\blip2-main\blip2-main\data\Flickr8k"

    clip_model_path = r"E:\doc\lhf\hf_models\clip-vit-base-patch32"
    opt_model_path = r"E:\doc\lhf\hf_models\opt-125m"

    checkpoint_path = r"E:\doc\lhf\blip2-main\blip2-main\outputs\checkpoints\mini_blip2_epoch_3.pt"

    output_dir = r"E:\doc\lhf\blip2-main\blip2-main\outputs"
    result_path = os.path.join(output_dir, "generate_results.txt")

    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 2. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # =========================
    # 3. 加载 processor 和 tokenizer
    # =========================
    processor = CLIPProcessor.from_pretrained(
        clip_model_path,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        opt_model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # 4. 加载 checkpoint
    # =========================
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    num_query_tokens = checkpoint.get("num_query_tokens", 8)
    qformer_layers = checkpoint.get("qformer_layers", 1)
    qformer_heads = checkpoint.get("qformer_heads", 8)
    max_length = checkpoint.get("max_length", 32)

    print("checkpoint loaded from:", checkpoint_path)
    print("checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("checkpoint avg_loss:", checkpoint.get("avg_loss", "unknown"))

    # =========================
    # 5. 加载数据
    # =========================
    dataset = Flickr8kCaptionDataset(
        root_dir=root_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_images=200,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # =========================
    # 6. 加载模型
    # =========================
    model = MiniBLIP2(
        vision_model_path=clip_model_path,
        language_model_path=opt_model_path,
        num_query_tokens=num_query_tokens,
        qformer_layers=qformer_layers,
        qformer_heads=qformer_heads,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # =========================
    # 7. 生成 15 个样例
    # =========================
    results = []

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Mini-BLIP2 Caption Generation Results\n")
        f.write("=" * 60 + "\n\n")

        for idx, batch in enumerate(dataloader):
            if idx >= 15:
                break

            pixel_values = batch["pixel_values"].to(device)

            gt_caption = batch["caption"][0]
            image_path = batch["image_path"][0]

            generated_captions = model.generate_caption(
                pixel_values=pixel_values,
                tokenizer=tokenizer,
                max_new_tokens=20,
                min_new_tokens=3,
                num_beams=3,
                prompt="A photo of",
            )

            pred_caption = generated_captions[0]

            result = {
                "index": idx + 1,
                "image_path": image_path,
                "ground_truth": gt_caption,
                "generated": pred_caption,
            }

            results.append(result)

            print("=" * 60)
            print(f"Sample {idx + 1}")
            print("Image:", image_path)
            print("Ground Truth:", gt_caption)
            print("Generated:", pred_caption)

            f.write(f"Sample {idx + 1}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Ground Truth: {gt_caption}\n")
            f.write(f"Generated: {pred_caption}\n")
            f.write("-" * 60 + "\n\n")

    print("\nGeneration finished.")
    print("results saved to:", result_path)


if __name__ == "__main__":
    generate()