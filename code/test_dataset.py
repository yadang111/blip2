from transformers import CLIPProcessor, AutoTokenizer
from torch.utils.data import DataLoader

from dataset import Flickr8kCaptionDataset


root_dir = r"E:\doc\lhf\blip2-main\blip2-main\data\Flickr8k"

clip_model_path = r"E:\doc\lhf\hf_models\clip-vit-base-patch32"
opt_model_path = r"E:\doc\lhf\hf_models\opt-125m"

processor = CLIPProcessor.from_pretrained(
    clip_model_path,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    opt_model_path,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = Flickr8kCaptionDataset(
    root_dir=root_dir,
    processor=processor,
    tokenizer=tokenizer,
    max_images=200,
    max_length=32
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

batch = next(iter(loader))

print("pixel_values:", batch["pixel_values"].shape)
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("caption example:", batch["caption"][0])
print("image path:", batch["image_path"][0])