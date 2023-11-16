#architecture of transformers
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

class AttentionMechanism:
    def __init__(self):
        self.weights = None

    def calculate_attention(self, query, key, value):
        score = query.dot(key)
        attention_weights = softmax(score)
        self.weights = attention_weights
        output = attention_weights.dot(value)
        return output


"""

The attention mechanism in transformers allows the model to focus on specific parts of input sequences when making predictions.
It assigns different weights to different words, capturing relationships and dependencies between them.
This dynamic attention mechanism enhances the model's ability to handle long-range dependencies in data.

"""
