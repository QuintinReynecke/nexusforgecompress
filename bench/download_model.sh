#!/bin/bash
# Download synthetic or small HF model subset
wget https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/model.safetensors -O test.safetensors
sha256sum test.safetensors  # For verification