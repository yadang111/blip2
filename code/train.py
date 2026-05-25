import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, AutoTokenizer
from torch.optim import AdamW

from dataset import Flickr8kCaptionDataset
from model import MiniBLIP2


def train():
    # =========================
    # 1. 路径配置
    # =========================
    root_dir = r"E:\doc\lhf\blip2-main\blip2-main\data\Flickr8k"

    clip_model_path = r"E:\doc\lhf\hf_models\clip-vit-base-patch32"
    opt_model_path = r"E:\doc\lhf\hf_models\opt-125m"

    output_dir = r"E:\doc\lhf\blip2-main\blip2-main\outputs"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_path = os.path.join(output_dir, "train_log.txt")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================
    # 2. 训练参数
    # =========================
    batch_size = 1
    num_epochs = 3
    learning_rate = 1e-4
    max_length = 32
    max_images = 200

    num_query_tokens = 16
    qformer_layers = 2
    qformer_heads = 8

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
    # 4. 加载数据集
    # =========================
    dataset = Flickr8kCaptionDataset(
        root_dir=root_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_images=max_images,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    print("dataset size:", len(dataset))

    # =========================
    # 5. 加载模型
    # =========================
    model = MiniBLIP2(
        vision_model_path=clip_model_path,
        language_model_path=opt_model_path,
        num_query_tokens=num_query_tokens,
        qformer_layers=qformer_layers,
        qformer_heads=qformer_heads,
    )

    model.to(device)
    model.train()

    model.print_trainable_parameters()

    # 只优化 requires_grad=True 的参数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # =========================
    # 6. 写入日志头
    # =========================
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Mini-BLIP2 Training Log\n")
        f.write(f"device: {device}\n")
        f.write(f"dataset size: {len(dataset)}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"epochs: {num_epochs}\n")
        f.write(f"learning rate: {learning_rate}\n")
        f.write(f"max length: {max_length}\n")
        f.write(f"num query tokens: {num_query_tokens}\n")
        f.write(f"qformer layers: {qformer_layers}\n")
        f.write("=" * 60 + "\n")

    # =========================
    # 7. 训练循环
    # =========================
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            global_step += 1

            if (step + 1) % 10 == 0 or step == 0:
                log_msg = (
                    f"epoch: {epoch + 1}, "
                    f"step: {step + 1}/{len(dataloader)}, "
                    f"global_step: {global_step}, "
                    f"loss: {loss_value:.4f}"
                )

                print(log_msg)

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")

        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time

        epoch_msg = (
            f"Epoch {epoch + 1} finished, "
            f"avg_loss: {avg_loss:.4f}, "
            f"time: {epoch_time:.2f}s"
        )

        print(epoch_msg)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

        # =========================
        # 8. 保存 checkpoint
        # =========================
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"mini_blip2_epoch_{epoch + 1}.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "num_query_tokens": num_query_tokens,
                "qformer_layers": qformer_layers,
                "qformer_heads": qformer_heads,
                "max_length": max_length,
            },
            checkpoint_path,
        )

        print("checkpoint saved to:", checkpoint_path)

    print("\nTraining finished.")
    print("log saved to:", log_path)


if __name__ == "__main__":
    train()