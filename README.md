# Chatbot with PDF Content Retrieval

This project includes Python code to address four different questions related to natural language processing and machine learning.

## Table of Contents

1. [Attention Mechanism](#attention-mechanism)
2. [RLHF vs Instruction Fine-Tuning](#rlhf-vs-instruction-fine-tuning)
3. [Parameter Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
4. [PDF-based Chatbot](#pdf-based-chatbot)

## Attention Mechanism

The `AttentionMechanism` class provides a simple implementation of an attention mechanism commonly used in transformer architectures.

- **File:** `attention_mechanism.py`

## RLHF vs Instruction Fine-Tuning

The `RLHFvsInstructionFineTuning` class compares two machine learning training approaches: Reinforcement Learning from Human Feedback (RLHF) and Instruction Fine-Tuning.

- **File:** `rlhf_vs_instruction_fine_tuning.py`

## Parameter Efficient Fine-Tuning

The `ParameterEfficientFineTuning` class explores various methods for parameter-efficient fine-tuning, such as knowledge distillation, pruning, and quantization.

- **File:** `parameter_efficient_fine_tuning.py`

## PDF-based Chatbot

The `PDFChatbot` class implements a chatbot that can answer questions based on the content of a given PDF file. It uses a vector store to parse and store the document.

- **Files:** 
    - `pdf_chatbot.py`
    - `vector_store_utils.py`

### How to Use

1. Install the required libraries: `pip install -r requirements.txt`
2. Replace `'pdfname.pdf'` in the example usage with the path to your PDF file.
3. Run the example script.

```bash
python pdf_chatbot.py
