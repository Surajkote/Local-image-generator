import torch
from diffusers import StableDiffusionPipeline

# Choose device: M1 GPU if available, else CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load Stable Diffusion 1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir="models",       # keep model inside project folder
    safety_checker=None,      # disable NSFW black-image filter
    torch_dtype=torch.float32 # use fp32 for stability on M1
).to(DEVICE)

# Reduce memory usage a bit
pipe.enable_attention_slicing()

prompt = input("Enter prompt: ")

image = pipe(
    prompt,
    num_inference_steps=30,  # more steps = better quality, slower
    height=384,              # keep 256x256 to avoid OOM
    width=384
).images[0]

image.save("output.png")
print("Saved output.png")
