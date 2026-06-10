#!/usr/bin/env python3
import os
import sys
import time

# ANSI Terminal Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    banner = f"""
{BLUE}{BOLD}======================================================================
     ____  ______ ______ ____   _____ ______ ______ ____  
    / __ \\/ ____// ____// __ \\ / ___// ____// ____// __ \\ 
   / / / / __/  / __/  / /_/ / \\__ \\/ __/  / __/  / / / /
  / /_/ / /___ / /___ / ____/ ___/ / /___ / /___ / /_/ / 
 /_____/_____//_____//_/     /____/_____//_____//_____/  
                                                         
          🚀 DEEPSEED - FINE-TUNING DECISION TOOL 🚀
======================================================================{END}
"""
    print(banner)

def get_choice(question, options):
    print(f"\n{BOLD}{question}{END}")
    for idx, opt in enumerate(options, 1):
        print(f"  [{idx}] {opt}")
    
    while True:
        try:
            choice = input(f"\nSelect an option (1-{len(options)}): ").strip()
            val = int(choice)
            if 1 <= val <= len(options):
                return val
            else:
                print(f"{RED}Invalid option. Please enter a number between 1 and {len(options)}.{END}")
        except ValueError:
            print(f"{RED}Invalid input. Please enter a valid number.{END}")

def main():
    clear_screen()
    print_banner()
    print("Welcome to the DeepSeed decision-support system.")
    print("This interactive tool evaluates your project constraints and suggests whether")
    print(f"you should use {GREEN}Prompt Engineering{END}, {BLUE}RAG{END}, or {YELLOW}Fine-Tuning{END}.\n")
    input("Press Enter to start the evaluation...")

    # Scoring initialized
    scores = {"Prompting": 0, "RAG": 0, "Fine-Tuning": 0}
    explanations = []

    # Question 1: Dataset size
    q1_opts = [
        "I have no training data, or fewer than 100 examples.",
        "I have a small labeled dataset (100 - 500 examples).",
        "I have a substantial, high-quality dataset (500 - 5,000 examples).",
        "I have a massive proprietary dataset (5,000+ examples) and resources to clean it."
    ]
    ans1 = get_choice("1. How much labeled training data do you currently have (or can realistically curate)?", q1_opts)
    if ans1 == 1:
        scores["Prompting"] += 4
        scores["RAG"] += 2
        scores["Fine-Tuning"] -= 3
        explanations.append("- Dataset size (< 100 examples) is highly insufficient for fine-tuning. Fine-tuning on a tiny dataset will cause severe overfitting.")
    elif ans1 == 2:
        scores["Prompting"] += 3
        scores["RAG"] += 2
        scores["Fine-Tuning"] += 1
        explanations.append("- With 100-500 examples, prompting remains strong, but you could attempt light classification fine-tuning if necessary.")
    elif ans1 == 3:
        scores["Fine-Tuning"] += 4
        scores["RAG"] += 2
        scores["Prompting"] += 1
        explanations.append("- With 500-5,000 examples, you are in the sweet spot for instruction fine-tuning (SFT) or parameter-efficient fine-tuning (LoRA).")
    else:
        scores["Fine-Tuning"] += 5
        scores["RAG"] += 1
        explanations.append("- A dataset size of 5,000+ examples is perfect for robust fine-tuning, allowing the model to adapt deeply to complex patterns.")

    # Question 2: Task Type
    q2_opts = [
        "General reasoning, brainstorming, coding, or common language tasks.",
        "Fact retrieval from private files (e.g. policy documents, code repositories, wikis).",
        "Enforcing strict output structures (e.g. valid JSON formatting, medical logs) and narrow style.",
        "Learning specialized language/vocabulary (e.g., specific medical terminology, legal codices)."
    ]
    ans2 = get_choice("2. What is the primary nature of the task?", q2_opts)
    if ans2 == 1:
        scores["Prompting"] += 5
        scores["Fine-Tuning"] -= 1
        explanations.append("- General reasoning and brainstorming are pre-trained capabilities. Fine-tuning is rarely beneficial here and may degrade reasoning.")
    elif ans2 == 2:
        scores["RAG"] += 5
        scores["Fine-Tuning"] += 1
        scores["Prompting"] += 1
        explanations.append("- Fact retrieval from documents requires dynamic context injection (RAG). Fine-tuning on facts is a bad practice as LLM weights are poor databases.")
    elif ans2 == 3:
        scores["Fine-Tuning"] += 5
        scores["Prompting"] += 1
        explanations.append("- Strict structural output (e.g., JSON schemas) is best handled by fine-tuning, which hardwires the structural rules into the attention layers.")
    else:
        scores["Fine-Tuning"] += 5
        scores["RAG"] += 2
        explanations.append("- Learning specialized domain vocabularies and writing styles is a primary use case for domain-specific fine-tuning.")

    # Question 3: Knowledge Updates
    q3_opts = [
        "Dynamic (changes hourly, daily, or weekly - e.g., prices, inventory, news).",
        "Semi-static (changes every few months - e.g., company policies, codebase versions).",
        "Completely static (changes rarely or never - e.g., medical diagnoses, formatting conventions)."
    ]
    ans3 = get_choice("3. How frequently does the core information/knowledge update?", q3_opts)
    if ans3 == 1:
        scores["RAG"] += 5
        scores["Prompting"] += 3
        scores["Fine-Tuning"] -= 4
        explanations.append("- Dynamic data is incompatible with fine-tuning due to training latency and compute cost. RAG is required to fetch real-time updates.")
    elif ans3 == 2:
        scores["RAG"] += 4
        scores["Fine-Tuning"] += 2
        explanations.append("- Semi-static data can use RAG or a combined RAG + periodic fine-tuning strategy.")
    else:
        scores["Fine-Tuning"] += 5
        scores["Prompting"] += 2
        explanations.append("- Static knowledge means training weights is highly durable. Once trained, the model retains the format and vocabulary indefinitely.")

    # Question 4: Latency & Cost Constraints
    q4_opts = [
        "Inference cost and latency are not concerns (running standard web API calls is fine).",
        "We need low latency, but token costs must be minimal (e.g., high-throughput production API).",
        "Strict local execution required (data cannot leave local servers due to HIPAA/compliance, low VRAM hardware)."
    ]
    ans4 = get_choice("4. What are your budget, latency, and hardware constraints?", q4_opts)
    if ans4 == 1:
        scores["Prompting"] += 4
        scores["RAG"] += 3
        scores["Fine-Tuning"] += 1
        explanations.append("- Flexible constraints favor API prompting or managed RAG, as they require no infrastructure setup.")
    elif ans4 == 2:
        scores["Fine-Tuning"] += 4
        scores["Prompting"] -= 2
        explanations.append("- High throughput benefits from fine-tuned smaller models (7B/8B). A 7B model has a much shorter prompt than a 32k prompt-engineered model, reducing inference cost.")
    else:
        scores["Fine-Tuning"] += 5
        scores["Prompting"] -= 3
        explanations.append("- Compliance and local execution require self-hosting open-weight models. Fine-tuning a local 8B model ensures data privacy and runs efficiently on single local GPUs.")

    # Question 5: Tolerance for Hallucinations
    q5_opts = [
        "High (creative writing, summarization, general draft assistants).",
        "Moderate (wants accurate details but can handle minor adjustments/checks).",
        "Zero tolerance (answers must be referenced directly from sources with citations)."
    ]
    ans5 = get_choice("5. What is the project's tolerance for factual hallucinations?", q5_opts)
    if ans5 == 1:
        scores["Prompting"] += 4
        scores["Fine-Tuning"] += 3
    elif ans5 == 2:
        scores["Prompting"] += 2
        scores["RAG"] += 3
        scores["Fine-Tuning"] += 2
    else:
        scores["RAG"] += 5
        scores["Fine-Tuning"] -= 2
        explanations.append("- Fact-critical systems requiring citations MUST use RAG. Fine-tuned models still hallucinate details and cannot provide native database citations.")

    # Evaluate results
    clear_screen()
    print_banner()
    print(f"{BOLD}=== EVALUATION COMPLETE ==={END}\n")

    # Normalize scores slightly to make them look nice (positive ranges)
    min_score = min(scores.values())
    normalized_scores = {k: max(0, v - min_score + 1) for k, v in scores.items()}
    total = sum(normalized_scores.values())
    percentages = {k: int((v / total) * 100) for k, v in normalized_scores.items()}

    # Print percentages
    print(f"{BOLD}Recommended Approach Breakdown:{END}")
    print(f"  {GREEN}Prompt Engineering:{END} {percentages['Prompting']}%")
    print(f"  {BLUE}RAG (Retrieval-Augmented):{END} {percentages['RAG']}%")
    print(f"  {YELLOW}Fine-Tuning:{END} {percentages['Fine-Tuning']}%")
    print("-" * 50)

    # Determine primary recommendation
    recommendation = max(scores, key=scores.get)
    rec_color = GREEN if recommendation == "Prompting" else (BLUE if recommendation == "RAG" else YELLOW)

    print(f"\n{BOLD}PRIMARY RECOMMENDATION: {rec_color}{recommendation.upper()}{END}\n")

    # Rationale summary
    print(f"{BOLD}Key Architectural Rationale:{END}")
    for exp in explanations:
        print(exp)
    
    # Save Report
    report_filename = "fine_tuning_feasibility_report.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"""# Project Feasibility Report: Architecture Evaluation

This report evaluates the optimal architecture for your language model project based on your requirements and constraints.

## Score Summary
* **Prompt Engineering:** {percentages['Prompting']}%
* **Retrieval-Augmented Generation (RAG):** {percentages['RAG']}%
* **Fine-Tuning:** {percentages['Fine-Tuning']}%

**Primary Recommendation:** {recommendation.upper()}

---

## Technical Rationale
{chr(10).join(explanations)}

---

## Architectural Deep Dive

### Option 1: Prompt Engineering (In-Context Learning)
* **When to use:** Prototyping, tasks with zero training data, general-purpose reasoning.
* **Pros:** Instantly deployable, zero compute/infrastructure overhead.
* **Cons:** High token consumption over time, strict output formats can slip under stress.

### Option 2: Retrieval-Augmented Generation (RAG)
* **When to use:** Factual question-answering over dynamic or proprietary documents.
* **Pros:** Prevents hallucinations, provides source citations, easy to update knowledge bases.
* **Cons:** Introduces multi-step pipeline latency, limited control over formatting and stylistic tone.

### Option 3: Fine-Tuning (Weight Adjustment)
* **When to use:** Specialized tasks, strict structural requirements, domain terminology, high token-cost reduction.
* **Pros:** Natively outputs structured data (JSON, code), decreases prompt length, runs locally on cheap hardware.
* **Cons:** Requires clean dataset preparation, high upfront time/compute cost, poor factual recall repository.
""")

    print(f"\n{GREEN}✔ A detailed Markdown feasibility report has been saved to: {UNDERLINE}{report_filename}{END}\n")

if __name__ == "__main__":
    main()
