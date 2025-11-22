from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir="models",  # keeps model inside project folder
    torch_dtype=torch.float32,
)
print("Model downloaded to models/")
