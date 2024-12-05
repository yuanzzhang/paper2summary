from collections import defaultdict
import spacy
import numpy as np
from rouge_score import rouge_scorer


class SpacyTokenizer:
    def __init__(self):
        self.tokenizer = spacy.load("en_core_web_sm").tokenizer

    def tokenize(self, text):
        return [token.text for token in self.tokenizer(text)]


TOKENIZER = SpacyTokenizer()
METRICS = ['rouge1', 'rouge2', 'rouge3', 'rougeL']


def evaluate_rouge(abstract, summary):
    scorer = rouge_scorer.RougeScorer(
        METRICS, 
        use_stemmer=True,
        tokenizer=TOKENIZER
    )

    scores = scorer.score(abstract.lower(), summary.lower())
    return {
        metric: score.fmeasure 
        for metric, score in scores.items()
    }


def aggregate_scores(all_scores):
    return {
        metric: 100 * sum(score[metric] for score in all_scores) / len(all_scores)
        for metric in METRICS
    }