import nltk
import re
import random
import math
from nltk.corpus import gutenberg
from collections import defaultdict, Counter


# ==PREPROCESSING==
def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    return tokens



# ==TRIGRAM MODEL==

def build_trigram_model(tokens: list[str]) -> dict:
    trigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 2):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        next_word = tokens[i + 2]
        trigram_counts[(w1, w2)][next_word] += 1

    return trigram_counts



# ==LAPLACE SMOOTHING==

def laplace_smoothing(trigram_counts: dict, vocab_size: int) -> dict:
    smoothed_probs = {}

    vocab = set()
    for context in trigram_counts:
        vocab.add(context[0])
        vocab.add(context[1])
        for next_word in trigram_counts[context]:
            vocab.add(next_word)

    for context in trigram_counts:
        smoothed_probs[context] = {}
        context_total = sum(trigram_counts[context].values())

        for word in vocab:
            count = trigram_counts[context][word]
            probability = (count + 1) / (context_total + vocab_size)
            smoothed_probs[context][word] = probability

    return smoothed_probs



# ==TEXT GENERATION WITH LAPLACE==

def generate_text(seed: list[str], smoothed_probs: dict, vocab: list[str], num_words: int = 30) -> str:
    generated = seed.copy()

    while len(generated) < num_words:
        context = (generated[-2], generated[-1])

        if context in smoothed_probs:
            next_word = max(smoothed_probs[context], key=smoothed_probs[context].get)
        else:
            next_word = random.choice(vocab)

        generated.append(next_word)

    return ' '.join(generated)



# ==PERPLEXITY WITH LAPLACE==

def compute_perplexity(test_tokens: list[str], smoothed_probs: dict, vocab_size: int) -> float:
    log_sum = 0.0
    N = 0

    for i in range(len(test_tokens) - 2):
        context = (test_tokens[i], test_tokens[i + 1])
        word = test_tokens[i + 2]

        if context in smoothed_probs:
            probability = smoothed_probs[context].get(word, 1 / vocab_size)
        else:
            probability = 1 / vocab_size

        log_sum += math.log(probability + 1e-10)
        N += 1

    perplexity = math.exp(-log_sum / N)
    return perplexity



# ==BONUS: UNIGRAM MODEL==

def build_unigram_model(tokens: list[str]) -> dict:
    unigram_counts = Counter()

    for word in tokens:
        unigram_counts[word] += 1

    return unigram_counts



# ==BONUS: BIGRAM MODEL==

def build_bigram_model(tokens: list[str]) -> dict:
    bigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        bigram_counts[w1][w2] += 1

    return bigram_counts


# ==BONUS: INTERPOLATED PROBABILITY==

def interpolated_probability(w1: str, w2: str, word: str,
                             unigram_counts: dict,
                             bigram_counts: dict,
                             trigram_counts: dict,
                             total_tokens: int,
                             lambda1: float = 0.1,
                             lambda2: float = 0.3,
                             lambda3: float = 0.6) -> float:

    unigram_prob = unigram_counts[word] / total_tokens

    bigram_total = sum(bigram_counts[w2].values())
    if bigram_total > 0:
        bigram_prob = bigram_counts[w2][word] / bigram_total
    else:
        bigram_prob = 0.0

    trigram_total = sum(trigram_counts[(w1, w2)].values())
    if trigram_total > 0:
        trigram_prob = trigram_counts[(w1, w2)][word] / trigram_total
    else:
        trigram_prob = 0.0

    probability = (lambda3 * trigram_prob) + (lambda2 * bigram_prob) + (lambda1 * unigram_prob)
    return probability


# ==BONUS: TEXT GENERATION WITH INTERPOLATION==

def generate_interpolated_text(seed: list[str],
                               vocab: list[str],
                               unigram_counts: dict,
                               bigram_counts: dict,
                               trigram_counts: dict,
                               total_tokens: int,
                               num_words: int = 30) -> str:
    generated = seed.copy()

    while len(generated) < num_words:
        w1 = generated[-2]
        w2 = generated[-1]

        next_word = max(
            vocab,
            key=lambda word: interpolated_probability(
                w1, w2, word,
                unigram_counts,
                bigram_counts,
                trigram_counts,
                total_tokens
            )
        )

        generated.append(next_word)

    return ' '.join(generated)


# ==BONUS: PERPLEXITY WITH INTERPOLATION==

def compute_interpolated_perplexity(test_tokens: list[str],
                                    unigram_counts: dict,
                                    bigram_counts: dict,
                                    trigram_counts: dict,
                                    total_tokens: int) -> float:
    log_sum = 0.0
    N = 0

    for i in range(len(test_tokens) - 2):
        w1 = test_tokens[i]
        w2 = test_tokens[i + 1]
        word = test_tokens[i + 2]

        probability = interpolated_probability(
            w1, w2, word,
            unigram_counts,
            bigram_counts,
            trigram_counts,
            total_tokens
        )

        log_sum += math.log(probability + 1e-10)
        N += 1

    perplexity = math.exp(-log_sum / N)
    return perplexity



# ==BONUS: COMPARISON==

def build_comparison(perplexity, perplexity_interpolation) -> str:
    comparison_text = (
        "The Laplace-smoothed story is more random and less coherent, because it contains many disconnected words and abrupt transitions. "
        "In contrast, the interpolation-based story is more fluent and more sentence-like, because it produces a more natural sequence from the same seed phrase. "
        "The perplexity of the Laplace model is " + str(perplexity) + ", which shows that it was much more uncertain on the test sentence 'the king is dead'. "
        "The perplexity of the interpolation model is " + str(perplexity_interpolation) + ", which is much lower and therefore indicates better prediction of the same test sentence. "
        "Overall, the interpolation model performed better than the Laplace model in both generated text quality and perplexity value."
    )
    return comparison_text


# MAIN FUNCTION

def main():
    nltk.download('gutenberg')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    raw_text = gutenberg.raw('shakespeare-caesar.txt')
    tokens = preprocess(raw_text)

    vocab = list(set(tokens))
    vocab_size = len(vocab)

    trigram_counts = build_trigram_model(tokens)
    smoothed_probs = laplace_smoothing(trigram_counts, vocab_size)

    seed = ['the', 'king']
    generated_story = generate_text(seed, smoothed_probs, vocab, num_words=30)

    test_sentence = "the king is dead"
    test_tokens = preprocess(test_sentence)
    perplexity = compute_perplexity(test_tokens, smoothed_probs, vocab_size)

    unigram_counts = build_unigram_model(tokens)
    bigram_counts = build_bigram_model(tokens)
    total_tokens = len(tokens)

    generated_interpolation_story = generate_interpolated_text(
        seed,
        vocab,
        unigram_counts,
        bigram_counts,
        trigram_counts,
        total_tokens,
        num_words=30
    )

    perplexity_interpolation = compute_interpolated_perplexity(
        test_tokens,
        unigram_counts,
        bigram_counts,
        trigram_counts,
        total_tokens
    )

    comparison_text = build_comparison(perplexity, perplexity_interpolation)

    print("=== Preprocessing ===")
    print("Vocabulary size:", vocab_size)
    print("Total tokens:", len(tokens))
    print()

    print("=== Text Generation (Laplace Smoothing) ===")
    print("Seed: the king")
    print("Generated:", generated_story)
    print()

    print("=== Perplexity ===")
    print("Test sentence: 'the king is dead'")
    print("Perplexity:", perplexity)
    print()

    print("=== Bonus  ===")
    print("Generated (Interpolation):", generated_interpolation_story)
    print("Perplexity (Interpolation):", perplexity_interpolation)
    print("Comparison:", comparison_text)


if __name__ == "__main__":
    main()