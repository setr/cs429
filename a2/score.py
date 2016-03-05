""" Assignment 2
"""
import abc
from collections import defaultdict
import math

import index


def idf(term, index):
    """ Compute the inverse document frequency of a term according to the
    index. IDF(T) = log10(N / df_t), where N is the total number of documents
    in the index and df_t is the total number of documents that contain term
    t.

    Params:
      terms....A string representing a term.
      index....A Index object.
    Returns:
      The idf value.

    >>> idx = index.Index(['a b c a', 'c d e', 'c e f'])
    >>> idf('a', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('d', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('e', idx) # doctest:+ELLIPSIS
    0.176...
    """
    return math.log10( len(index.documents) / index.doc_freqs[term] )

class ScoringFunction:
    """ An Abstract Base Class for ranking documents by relevance to a
    query. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, query_vector, index):
        """
        Do not modify.

        Params:
          query_vector...dict mapping query term to weight.
          index..........Index object.
        """
        return


class RSV(ScoringFunction):
    """
    See lecture notes for definition of RSV.

    idf(a) = log10(3/1)
    idf(d) = log10(3/1)
    idf(e) = log10(3/2)
    >>> idx = index.Index(['a b c', 'c d e', 'c e f'])
    >>> rsv = RSV()
    >>> rsv.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.4771...
    """
    #sum of the log of inverse document frequency for each matching query term
    # * weight
    # weight = ((k + 1) * term-freq) / (k + term-freq)

    def score(self, query_vector, index):
        # get a list of query terms, and their idf vals?
        # doc_id = index
        # idf val = value
        # get the documents with more than 1?

        # make a dictionary of RSV(d)
        # doc : RSV(val)
        # take a doc
        # get its query val
        # 

        doclist = list()

        for term, k in query_vector.items():
            idf_val = math.log10(idf(term, index))
            for i, doc in enumerate(index.documents):
                tf = doc.count(term)
                weight = ((k + 1) * tf) / (k + tf)
                RSV = idf_val*weight
                if len(doclist[i]):
                    doclist[i] += RSV
                else:
                    doclist[i] = RSV
        return doclist


    def __repr__(self):
        return 'RSV'


class BM25(ScoringFunction):
    """
    See lecture notes for definition of BM25.

    log10(3) * (2*2) / (1(.5 + .5(4/3.333)) + 2) = log10(3) * 4 / 3.1 = .6156...
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> bm = BM25(k=1, b=.5)
    >>> bm.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.61564032...
    """
    def __init__(self, k=1, b=.5):
        self.k = k
        self.b = b

    def score(self, query_vector, index):
        ###TODO
        pass

    def __repr__(self):
        return 'BM25 k=%d b=%.2f' % (self.k, self.b)


class Cosine(ScoringFunction):
    """
    See lecture notes for definition of Cosine similarity.  Be sure to use the
    precomputed document norms (in index), rather than recomputing them for
    each query.

    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> cos = Cosine()
    >>> cos.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.792857...
    """
    # cosine similarity is the similarity between two documents
    # dot product doc1 x doc2
    # divided by (normalization of doc1 * normalization of doc2)

    # doc1 = query
    # doc2 = target-doc
    def score(self, query_vector, index):
        scores = defaultdict(lambda: 0)

       #The first document is ['a a b c']. So, the term a has tf=2 and df=1; thus, its tf-idf value is (1 + log10(2)) * log(3/1) = .6207.... We multiply this by the provided query weight for this term (1). Thus, the numerator in the cosine similarity is .6207.... The denominator is the precomputed doc norm for document 1 (0.782927). Then, the final value is .6207... / .782927 = 0.792857. Recall that we don't need to divide by the query norm, since this will not affect the final ranking.         
        for q_term, q_weight in query_vector.items():
            for doc_id, doc_tf in index.index[q_term]:
                tf = doc_tf 
                df = index.doc_freqs[q_term]
                tf_idf = (1.0 + math.log10(doc_tf)) * math.log10(len(index.documents) / df)
                numerator = tf_idf * q_weight
                scores[doc_id] += numerator

        for doc_id in scores:
            scores[doc_id] /= index.doc_norms[doc_id]
        return scores

    def __repr__(self):
        return 'Cosine'
