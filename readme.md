# LLM Course Environment Setup

This guide walks you through creating a dedicated Python environment and installing all required packages for the Large Language Models (LLM) module.

## Prerequisites

Before you begin, ensure you have:

* Python 3.11
* Conda (Anaconda or Miniconda)
* Pip

---

# 1. Create a New Python Environment

To keep dependencies isolated from other projects, create a dedicated Conda environment for this course.

```bash
conda create --name llms_env python=3.11

conda activate llms_env
```

After activation, your terminal prompt should display:

```bash
(llms_env)
```

---

# 2. Install Required Packages

Install all required libraries using pip.

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install openai==0.28 config==0.5.1 langchain==0.0.297 pydantic==1.10.9 tiktoken==0.5.1 faiss-cpu==1.7.4 transformers==4.47.1 datasets==3.2.0 evaluate==0.4.3 accelerate==1.2.1 ipywidgets==8.1.5 matplotlib==3.10.0 seaborn==0.13.2 clean-text==0.6.0 scikit-learn==1.6.0 sentencepiece==0.2.0 pandas==2.0.0

```

---

# 3. Install Jupyter Support

To use this environment inside Jupyter Notebook or JupyterLab, install the following packages:

```bash
pip install ipykernel jupyterlab notebook
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=llms_env
```

When launching Jupyter Notebook or JupyterLab, select:

```text
llms_env
```

as your active kernel.

---

# Package Overview

The following libraries will be used throughout the LLM course.

## OpenAI

**Version:** 0.28

Official Python client for interacting with OpenAI GPT models and APIs.

```python
import openai
```

---

## Config

**Version:** 0.5.1

Lightweight configuration management library for storing and loading project settings.

---

## LangChain

**Version:** 0.0.297

Framework for building applications powered by Large Language Models.

Features include:

* Prompt chaining
* Agent workflows
* Retrieval-Augmented Generation (RAG)
* Vector database integrations
* Tool usage and orchestration

```python
from langchain import LLMChain
```

---

## Pydantic

**Version:** 1.10.9

Provides data validation and structured models for reliable LLM inputs and outputs.

```python
from pydantic import BaseModel
```

---

## Tiktoken

**Version:** 0.5.1

Tokenizer optimized for OpenAI models.

Useful for:

* Counting tokens
* Estimating API costs
* Splitting text into model-friendly chunks

---

## FAISS

**Version:** 1.7.4

Facebook AI Similarity Search (FAISS) enables fast vector similarity search.

Commonly used for:

* Semantic search
* Embedding retrieval
* Question-answering systems

---

## Transformers

**Version:** 4.47.1

Hugging Face library providing thousands of pretrained transformer models.

Examples:

* BERT
* GPT
* T5
* RoBERTa
* DistilBERT

```python
from transformers import AutoTokenizer
```

---

## PyTorch

**Version:** 2.5.1

Deep learning framework used to train and run neural networks.

Most modern NLP and LLM workflows rely on PyTorch.

```python
import torch
```

---

## Datasets

**Version:** 3.2.0

Provides access to large machine learning and NLP datasets.

Benefits include:

* Efficient loading
* Streaming support
* Built-in preprocessing

---

## Evaluate

**Version:** 0.4.3

Library for calculating machine learning evaluation metrics.

Examples:

* Accuracy
* Precision
* Recall
* F1 Score
* BLEU

---

## Accelerate

**Version:** 1.2.1

Simplifies model training across:

* CPUs
* GPUs
* Multiple GPUs
* Distributed environments

---

## ipywidgets

**Version:** 8.1.5

Adds interactive controls to Jupyter notebooks.

Examples:

* Sliders
* Buttons
* Dropdown menus
* Interactive parameter tuning

---

## Matplotlib

**Version:** 3.10.0

Core visualization library for Python.

Used for:

* Charts
* Graphs
* Training metrics
* Data exploration

---

## Seaborn

**Version:** 0.13.2

High-level statistical visualization library built on top of Matplotlib.

Provides cleaner and more informative visualizations with minimal code.

---

## Clean-Text

**Version:** 0.6.0

Utility library for cleaning and normalizing text.

Examples:

* Removing HTML
* Removing special characters
* Normalizing whitespace
* Preparing text for NLP pipelines

---

## Scikit-Learn

**Version:** 1.6.0

General-purpose machine learning library.

Useful for:

* Classification
* Clustering
* Feature engineering
* Model evaluation

```python
from sklearn.model_selection import train_test_split
```

---

## SentencePiece

**Version:** 0.2.0

Subword tokenizer commonly used in modern language models such as T5 and LLaMA.

Benefits:

* Language independent
* Handles unknown words effectively
* Supports efficient tokenization

---

## Pandas

**Version:** 2.0.0

Essential library for structured data manipulation and analysis.

```python
import pandas as pd
```

Common uses:

* Reading CSV files
* Data cleaning
* Exploratory data analysis
* Dataset preparation

---

# Verify Installation

Run the following command to verify your environment:

```bash
python -c "import openai, langchain, torch, transformers, pandas; print('Installation successful!')"
```

Expected output:

```text
Installation successful!
```

---

# Launch Jupyter

Start JupyterLab:

```bash
jupyter lab
```

Or start Jupyter Notebook:

```bash
jupyter notebook
```

Select the **llms_env** kernel before running any course notebooks.

---

## Ready to Go! 🚀

Your LLM course environment is now configured and ready for:

* Prompt Engineering
* OpenAI API Usage
* Embeddings
* Vector Databases
* Retrieval-Augmented Generation (RAG)
* Hugging Face Models
* Fine-Tuning Experiments
* LLM Evaluation
* Interactive Notebook Development

Happy Coding! 🎉