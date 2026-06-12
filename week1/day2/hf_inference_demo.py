#!/usr/bin/env python3
"""
DeepSeed - Week 1 Day 2: Hugging Face Ecosystem Demo
This script demonstrates how to load a model and tokenizer from the Hugging Face Hub,
apply a structured chat template, and run streaming inference locally.

Model Used: Qwen/Qwen2.5-1.5B-Instruct
(A highly capable, lightweight model that runs efficiently on consumer hardware).
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    # 1. Define Model ID
    # We use the 1.5B parameter Qwen model. It provides excellent instruction-following
    # quality while only requiring ~3GB of VRAM (or RAM if run on CPU).
    # You can change this to "Qwen/Qwen2.5-7B-Instruct" if you have a GPU with 16GB+ VRAM.
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("======================================================================")
    print(f"🚀 Loading Model & Tokenizer: {model_id}")
    print("======================================================================\n")

    # 2. Determine the optimal hardware device
    # GPU (CUDA) is preferred for fast inference. If not available, we fall back to CPU.
    if torch.cuda.is_available():
        device = "cuda"
        # BF16 (Bfloat16) is highly recommended for modern GPUs (Ampere/Ada Lovelace/Hopper).
        # We fall back to FP16 (Float16) for older GPUs, and FP32 for CPU.
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"✅ CUDA GPU detected. Using device: '{device}' with dtype: {torch_dtype}\n")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print(f"⚠️ CUDA GPU not detected. Using device: '{device}' (Warning: CPU inference will be slow!)\n")

    # 3. Load the Tokenizer
    # The tokenizer breaks down raw text into integer Token IDs (vocabulary index).
    print("📥 Downloading/Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("✔ Tokenizer loaded successfully.\n")

    # 4. Load the Model
    # AutoModelForCausalLM is used for autoregressive language models (decoder-only architectures).
    # device_map="auto" automatically splits the model across available GPUs if needed.
    # For CPU execution, we explicitly map to the device or let PyTorch handle it.
    print("📥 Downloading/Loading Model Weights (approx. 3GB)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
        print("✔ Model weights loaded successfully.\n")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("Please ensure you have installed the required dependencies:")
        print("pip install torch transformers accelerate")
        sys.exit(1)

    # 5. Define the Conversation Messages
    # Instruction-tuned models are trained on specific conversational structures.
    # We define our prompt as a list of message dictionaries (system prompt, user prompt).
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, enthusiastic, and highly technical assistant."
        },
        {
            "role": "user",
            "content": "Explain the relationship between the Hugging Face 'transformers' library and the 'peft' library in 3 bullet points."
        }
    ]

    # 6. Apply the Chat Template
    # Every model family (Llama, Qwen, Mistral) has a different formatting convention for chat.
    # Qwen uses ChatML. `apply_chat_template` automatically maps our messages list into the 
    # exact special control tokens required by Qwen.
    # add_generation_prompt=True appends the assistant token marker indicating the model should start generating.
    print("✍ Formatting prompt with model's Chat Template...")
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n--- Formatted ChatML Prompt ---")
    print(formatted_prompt.replace("\n", "\\n")) # Display special newline mappings
    print("--------------------------------\n")

    # 7. Tokenize the input text
    # Convert the formatted string into PyTorch tensors of Token IDs.
    # return_tensors="pt" returns PyTorch tensors.
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    # 8. Setup a Text Streamer
    # A streamer allows us to print tokens to the standard output in real-time as they 
    # are generated, rather than waiting for the entire sequence to finish.
    # skip_prompt=True avoids printing the user prompt back to us.
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 9. Generate Response
    # We call model.generate with standard generation hyperparameters.
    print(f"🔮 Generating response from {model_id} (Streaming):\n")
    print("--- Model Output ---")
    
    with torch.no_grad(): # Disable gradient calculations to save VRAM/compute during inference
        model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id # Set padding token to end-of-sequence token
        )
        
    print("\n--------------------")
    print("\n🎉 Inference complete! You have successfully run a model local inference today.")

if __name__ == "__main__":
    main()
