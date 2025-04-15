# %%
import torch
# torch used for the torchscript model, where GPU acceleration is needed, which helps to speed up the inference process

# %%
print("PyTorch version:", torch.__version__)
# Check if torch is compiled with CUDA, meaning it can use GPU acceleration
print("CUDA available:", torch.cuda.is_available())
# Check if torch backends are compiled with CUDA, and cuDNN is available in the system
print("cuDNN version:", torch.backends.cudnn.version())
# Check the number of GPUs available, this confirms GPUs are detected and available for use
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# %%
# Import pipeline, AutoTokenizer, and AutoModelForCausalLM from transformers
# pipeline : this is a high-level API for using models easily, it gives you a simple interface to use models for various tasks. example: text generation, translation, etc.
# AutoTokenizer : this is used to convert text into tokens that the model can understand. It handles the preprocessing of text data.
# AutoModelForCausalLM : this is a class for loading pre-trained models for causal language modeling tasks. It allows you to use models like GPT-2, GPT-3, etc.

# Import the BitsAndBytesConfig class for quantization. 
# BitsAndBytesConfig : this is used for configuring quantization settings for models. It helps in reducing the model size and improving inference speed without significant loss in performance.
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# %%
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the model with 4-bit quantization using the BitsAndBytesConfig class
# The model is loaded with 4-bit quantization to reduce memory usage and improve inference speed.
# load_in_4bit : this parameter indicates that the model should be loaded in 4-bit precision.
# bnb_4bit_use_double_quant : this parameter indicates whether to use double quantization for better performance. (Double quantization is a technique that helps in reducing the model size further while maintaining performance.)
# bnb_4bit_quant_type : this parameter specifies the type of quantization to be used. "nf4" is a specific quantization type that is optimized for performance.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model using the AutoModelForCausalLM class with the specified model ID and quantization configuration.
# device_map : this parameter indicates how to distribute the model across available devices. "auto" automatically places the model on available devices (CPU/GPU).
# The model is loaded with the specified quantization configuration to reduce memory usage and improve inference speed.
# This model is loaded in GPU memory and everytime we call the model, it will be loaded in GPU memory. It then uses the GPU for inference.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Automatically place model on available devices (CPU/GPU)
)

# Load the tokenizer using the AutoTokenizer class with the specified model ID.
# The tokenizer is responsible for converting text into tokens that the model can understand.
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# %% [markdown]
# 
# # **What the progress bar shows**
# 
# You're running:
# 
# ```python
# model = AutoModelForCausalLM.from_pretrained(...)
# ```
# 
# This line tells Hugging Face to:
# 
# 1. **Download the model weights and config** from the Hub.
# 2. **Quantize the model (4-bit in your case)** using `BitsAndBytesConfig`.
# 3. **Place it on the correct device** (`device_map="auto"` handles that).
# 
# ---
# 
# ### The progress bars:
# 
# #### `config.json`  
# - This file defines the model architecture and tokenizer setup.
# 
# ---
# 
# #### ⚙️ `model.safetensors.index.json`  
# - This tells Hugging Face how to **split and reference** the model weights.
# - Since these big models are often too large for a single file, they’re split into chunks (`model-00001`, `00002`, etc.).
# 
# ---
# 
# #### 📦 The 3 large files:
# - `model-00001-of-00003.safetensors`  
# - `model-00002-of-00003.safetensors`  
# - `model-00003-of-00003.safetensors`
# 
# Each of these is a **partial weight file** (chunks of the full model parameters):
# - You're downloading ~15 GB total (around 5 GB each).
# - The download is in progress, shows speed and ETA for each file.
# 
# ---
# 
# ### The warning:
# 
# ```txt
# UserWarning: 'huggingface_hub' cannot create symlinks...
# ```
# 
# It means:
# - On Windows, creating "symlinks" (shortcut-like references) is restricted unless you're in **Developer Mode** or running Python as an **Administrator**.
# - Hugging Face uses symlinks sometimes to save disk space.
# 
# ---
# 
# ### In Summary
# 
# You're successfully:
# - Loading the Mistral 7B Instruct model
# - With 4-bit quantization
# - While downloading its 3 weight shards
# 
# And you will be able to run inference!

# %% [markdown]
# #### Command to view for real time GPU utilization in Command Prompt `nvidia-smi -l 1`

# %% [markdown]
# #### Show total GPU memory reserved and allocated and current model loaded to GPU

# %%

# Allocated basically means the model is using this much memory
# Reserved means the memory is reserved for PyTorch for inference, but not all of it is being used by the model.
print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Reserved: ", torch.cuda.memory_reserved() / 1e9, "GB")

# %%
#Current Loaded model
print(model.__class__.__name__)


