"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):
    def __init__(self):
        self.vocab = set()
        self.class_terms = defaultdict(lambda: defaultdict(lambda: 1))
        self.class_prior = dict()


    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        return self.class_terms[label][term]

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        score = []
        for term in self.class_terms[label]:
           prob = self.class_terms[label][term] 
           not_c = sum([(self.class_terms[c][term] if term in self.class_terms[c] else 1) for c in self.class_terms if c != label])

           odds = (prob * 1.0) / (not_c * 1.0)
           score.append((odds, term))
        return sorted(score, reverse=True, key= lambda x: x[0])[:n]

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        # Need Vocabulary, dict of class:prior, dict of class: term: prob 
        # vocab = set
        # class_prior = {class : prior}
        # class_terms = {class : {term : probability}}

        class_index = defaultdict(list)
        # { class : [[tokens]] } # list of docs, with docs being a list of tokens.
        class_termcount = defaultdict(lambda: defaultdict(lambda:1))
        class_total = defaultdict(lambda: 1)
        for d in documents:
            class_index[d.label].append(d.tokens)
            self.vocab |= set(d.tokens) # |= union

            for token in d.tokens:
                class_termcount[d.label][token] += 1
                class_total[d.label] +=1

        for c in class_index:
            num_of_class = len(class_index)
            prior = (len(class_index[c]) * 1.0) / (len(documents) * 1.0)  # freq of class in collection

            # for every term that exists in the collection
            # term_count = the number of times that term appears in the class's docs
            # total = running count of term's appearance in the class
            prob = dict()
            for t in self.vocab:
                prob[t] = (class_termcount[c][t]) / (class_total[c]) 

            # now to just add everything to class globals
            self.class_prior[c] = prior
            self.class_terms[c] = prob


    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        label_list = list()
        for doc in documents:
            doc_vocab = {token for token in doc.tokens}
            score = dict()
            for c in self.class_terms:
                # log(P(c)) * sum([ log10(P(t|c)) ])
                score[c] = math.log10(self.class_prior[c])
                score[c] += sum([math.log10(self.class_terms[c][term] if term in self.class_terms[c] else 1) for term in doc_vocab])
            label, _ = max(score.items(), key=lambda x:x[1])
            label_list.append(label)
        return label_list

def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    total = len(documents)
    correct = 0
    num_ham = 0
    num_spam = 0
    for i in range(len(documents)):
        p = predictions[i]
        d = documents[i].label
        if p == d:
            correct += 1
        elif p == "ham":
            num_ham += 1
        else:
            num_spam += 1
    return ((correct/total), num_ham, num_spam)

def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
