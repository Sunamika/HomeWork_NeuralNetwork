# CS5720 – Home Assignment 4

## Assignment Overview

This assignment explores fundamental concepts in natural language processing and Transformer architectures, including sequence-to-sequence models, attention mechanisms, self-attention, multi-head attention, and positional encoding.

The programming section demonstrates three important NLP techniques using Python.

## Programming Tasks

### Q1: NLP Preprocessing Pipeline

Implemented text tokenization, English stopword removal, and Porter stemming using the NLTK library.

### Q2: Named Entity Recognition

Implemented Named Entity Recognition using spaCy to identify entities, their categories, and their character positions in a sentence.

### Q3: Scaled Dot-Product Attention

Implemented scaled dot-product attention using NumPy. The program calculates query-key similarity scores, scales them, applies softmax, and computes the final attention output.

## Technologies Used

- Python 3
- NumPy
- NLTK
- spaCy

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Execution

```bash
python Q1_NLP_Preprocessing.py
python Q2_Named_Entity_Recognition.py
python Q3_Scaled_Dot_Product_Attention.py
```

## Conclusion

This assignment provided practical experience with NLP preprocessing, named entity recognition, and scaled dot-product attention. These techniques are fundamental to understanding how modern Transformer-based language models process and represent textual information.
