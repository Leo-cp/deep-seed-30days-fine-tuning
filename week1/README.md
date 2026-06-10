# Week 1 — Foundation (Days 1–7)

Welcome to the first week of your 30-Day Fine-Tuning & Model Training journey. This week is focused on establishing core concepts, understanding the Hugging Face library, curating data, and running your first parameter-efficient fine-tuning (PEFT) tasks.

## 📅 Daily Schedule & Roadmap

- [ ] **Day 1: Why Fine-Tune?**
  - *Goal:* Master the decision matrix of when to fine-tune vs. prompt engineering/RAG.
  - *Key Concepts:* Specialist vs. Generalist, cost-efficiency (7B vs. GPT-4), formatting constraints, dataset thresholds (<500 examples).
  - *Deliverable:* Complete the decision evaluation tool in [`day1/`](file:///c:/Users/USER/Desktop/DEEPSEED/deep-seed-30days-fine-tuning/week1/day1).

- [ ] **Day 2: Hugging Face Ecosystem**
  - *Goal:* Navigate the Hugging Face Hub, model classes, and load your first large model.
  - *Key Concepts:* `transformers`, `datasets`, `peft`, `trl`, `accelerate`.
  - *Deliverable:* Load Qwen2.5-7B, run local inference, and set up your Hugging Face Hub account.

- [ ] **Day 3: Dataset Curation & Formatting**
  - *Goal:* Understand instruction-tuning formats and clean custom training datasets.
  - *Key Concepts:* Instruction formatting (`{"instruction": ..., "input": ..., "output": ...}`), MinHash deduplication, perplexity filtering.
  - *Deliverable:* Prepare a clean dataset of 2,000 instruction-tuning examples.

- [ ] **Day 4: LoRA: Low-Rank Adaptation**
  - *Goal:* Deep dive into the mathematics and implementation of Low-Rank Adaptation (PEFT).
  - *Key Concepts:* Matrix decomposition ($\Delta W = AB$), Rank ($r$), Alpha ($\alpha$), trainable vs. frozen weights.
  - *Deliverable:* Implement LoRA adapters using PyTorch and `peft`.

- [ ] **Day 5: QLoRA: Fine-Tuning on Consumer Hardware**
  - *Goal:* Fine-tune a 3B model (e.g., Llama-3.2-3B) in 4-bit precision.
  - *Key Concepts:* 4-bit quantization, `bitsandbytes`, double quantization, paging optimizers.
  - *Deliverable:* Run a QLoRA fine-tuning run on a single GPU.

- [ ] **Day 6: Unsloth: Fast Fine-Tuning**
  - *Goal:* Optimize fine-tuning speed and memory usage by 2-5x.
  - *Key Concepts:* Custom CUDA kernels, memory-efficient backpropagation.
  - *Deliverable:* Replicate the QLoRA experiment using Unsloth and compare VRAM/speed metrics.

- [ ] **Day 7: Sunday Presentation**
  - *Goal:* Consolidate and showcase your learnings.
  - *Deliverable:* Create a visual LoRA diagram and prepare 5 comparison test prompts (Base Model vs. Fine-Tuned Model).

---

## 🛠️ Prerequisites & Installation

To run the experiments and scripts for this week, we recommend setting up a virtual environment (Python 3.10+) and installing the core libraries:

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch (adjust based on CUDA availability, e.g., CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face ecosystem
pip install transformers datasets peft trl accelerate bitsandbytes
```

*Note: For Day 6 (Unsloth), specific installation commands using Unsloth's optimized packages will be provided.*
