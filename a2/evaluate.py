""" Assignment 2
"""
import abc

import numpy as np


class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.

        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        # % of hits that were relevant
        return len(set(hits) & set(relevant))  * 1.0 / len(hits)

    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.

        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        # % of relevant found
        return len(set(hits) & set(relevant))  * 1.0 / len(relevant)

    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.

        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        # weights precision and recall equally
        # (2 * precision * recall) / (precision + recall)
        precision = len(set(hits) & set(relevant))  * 1.0 / len(hits)
        recall = len(set(hits) & set(relevant))  * 1.0 / len(relevant)
        numer = 2 * precision * recall
        denom = precision + recall
        return numer / denom

    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.

        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """

        relevant_indexes = []
        for index, x in enumerate(hits):
            if x in relevant:
                relevant_indexes.append(index)

        t = 0
        for i in relevant_indexes:
            i = i+1
            precision = len(set(hits[:i]) & set(relevant))  * 1.0 / i
            # precision = sum(map(lambda x: 1 if x in relevant else 0, hits[:i])) * 1.0 / i
            t += precision
        return t / len(relevant)

    def __repr__(self):
        return 'MAP'

