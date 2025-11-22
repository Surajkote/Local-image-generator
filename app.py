import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# --------------------------
# Page UI Configuration
# --------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI polish
st.markdown("""
<style>
body {
    background-color: #121212;
}
.block-container {
    padding-top: 2rem;
}
.sidebar .sidebar-content {
    background: #1b1b1b;
}
h1, h2, h3, h4, h5, h6, label, p {
    color: #eaeaea !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_pipeline():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="models",
        safety_checker=None,
        torch_dtype=torch.float32
    ).to(device)

    pipe.enable_attention_slicing()
    return pipe


# --------------------------
# UI Layout
# --------------------------
st.title("🎨 Local AI Image Generator")
st.write("Generate high-quality images locally using **Stable Diffusion 1.5**.")

prompt = st.text_input(
    "Enter a detailed prompt",
    placeholder="Example: A futuristic cyberpunk sneaker product shot, neon lights, studio lighting"
)


# Sidebar Controls
with st.sidebar:
    st.header("⚙ Generation Settings")

    steps = st.slider("Diffusion Steps", 10, 40, 30)
    guidance = st.slider("Guidance Scale", 1.0, 12.0, 7.5)

    st.write("---")
    st.caption("⚠ Running on local Mac — keep resolution low")

    height = st.selectbox("Image Height", [256, 384, 512], index=0)
    width = st.selectbox("Image Width", [256, 384, 512], index=0)

generate_btn = st.button("🚀 Generate Image", use_container_width=True)


# --------------------------
# Generate Image
# --------------------------
if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        st.write("⏳ Loading model (first run may take a while)...")
        pipe = load_pipeline()

        with st.spinner("🎨 Generating image..."):
            result = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width
            )
            image = result.images[0]

        st.image(image, caption="Generated Image", use_column_width=True)

        # Save file
        image.save("output.png")
        st.success("Image saved as `output.png`")
