# N-gram Story Generator


This is a Natural Language Processing (NLP) project that implements an N-gram Language Model from scratch using Python. It learns statistical patterns from text data and generates a short story using unigram, bigram, and trigram probabilities.

## Overview

The model is trained on a text corpus (NLTK Gutenberg dataset) and uses trigram prediction as the primary model. It applies Laplace smoothing to handle unseen sequences and generates a 30-word story using a greedy approach.

The project also uses linear interpolation by combining unigram, bigram, and trigram probabilities to improve prediction quality and model performance.

## Features

- Text preprocessing (lowercasing, punctuation removal, tokenization)  
- Trigram language model built from scratch  
- Laplace smoothing implementation  
- Greedy text generation  
- Perplexity calculation  
- Linear interpolation using unigram, bigram, and trigram  
- Comparison between Laplace smoothing and interpolation  

## Dataset

This project uses the NLTK Gutenberg corpus (Shakespeare – Julius Caesar).

Make sure to download required NLTK resources before running the code.

## Installation

Install dependencies:

pip install nltk

Download NLTK data:

import nltk  
nltk.download('gutenberg')  
nltk.download('punkt')


## Output

The program prints:

- Vocabulary size and total tokens  
- Generated story using Laplace smoothing  
- Perplexity score  
- Generated story using interpolation  
- Interpolated perplexity  
- Comparison between both approaches  

## Example

Seed: the king  

Generated story (example):  
the king was a man who had no fear and spoke with great courage in the battle of rome and the people followed him  

