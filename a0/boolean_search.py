#!/usr/bin/python3
""" Assignment 0

You will implement a simple in-memory boolean search engine over the jokes
from http://web.hawkesnest.net/~jthens/laffytaffy/.

The documents are read from documents.txt.
The queries to be processed are read from queries.txt.

Your search engine will only need to support AND queries. A multi-word query
is assumed to be an AND of the words. E.g., the query "why because" should be
processed as "why AND because."
"""
from collections import defaultdict
import re


def tokenize(document):
    """ Convert a string representing one document into a list of
    words. Remove all punctuation and split on whitespace.
    Here is a doctest: 
    >>> tokenize("Hi  there. What's going on?")
    ['hi', 'there', 'what', 's', 'going', 'on']
    """
    # drop punctuation
    return re.findall('\w+', document.lower())
    #return re.split('\W+', document.lower())
    #document = re.sub(r'[\W\S]', '', document)  # replace not-word, not-space with ''
    # then split
    #return document.split()

def create_index(tokens):
    """
    Create an inverted index given a list of document tokens. The index maps
    each unique word to a list of document ids, sorted in increasing order.
    Params:
      tokens...A list of lists of strings
    Returns:
      An inverted index. This is a dict where keys are words and values are
      lists of document indices, sorted in increasing order.
    Below is an example, where the first document contains the tokens 'a' and
    'b', and the second document contains the tokens 'a' and 'c'.
    >>> index = create_index([['a', 'b'], ['a', 'c']])
    >>> sorted(index.keys())
    ['a', 'b', 'c']
    >>> index['a']
    [0, 1]
    >>> index['b']
    [0]
    """
    # input:
    # [['a','b']
    #  ['c','d']]
    index = {}
    for doc_id, token_list in enumerate(tokens):
        for token in token_list:
            if token in index:
                index[token].append(doc_id)
            else:
                index[token] = [doc_id]
    return index

def intersect(list1, list2):
    """ Return the intersection of two posting lists. Use the optimize
    algorithm of Figure 1.6 of the MRS text. Your implementation should be
    linear in the sizes of list1 and list2. That is, you should only loop once
    through each list.
    Params:
      list1....A list of document indices, sorted in ascending order.
      list2....Another list of document indices, sorted in ascending order.
    Returns:
      The list of document ids that appear in both lists, sorted in ascending order.
    >>> intersect([1, 3, 5], [3, 4, 5, 10])
    [3, 5]
    >>> intersect([1, 2], [3, 4])
    []
    """
    answer = []
    x = 0
    y = 0
    max1 = len(list1)
    max2 = len(list2)
    while x < max1 and y < max2:
        docid1 = list1[x]
        docid2 = list2[y]
        if docid1 == docid2:
            answer.append(docid1)
            x += 1
            y += 1
        elif docid1 < docid2:
            x += 1
        else:
            y += 1
    return answer


def sort_by_num_postings(words, index):
    """
    Sort the words in increasing order of the length of their postings list in
    index. You may use Python's builtin sorted method.
    Params:
      words....a list of strings.
      index....An inverted index; a dict mapping words to lists of document
      ids, sorted in ascending order.
    Returns:
      A list of words, sorted in ascending order by the number of document ids
      in the index.

    >>> sort_by_num_postings(['a', 'b', 'c'], {'a': [0, 1], 'b': [1, 2, 3], 'c': [4]})
    ['c', 'a', 'b']
    """
    # makes a set of (word, doc_count)
    word_counts = [(word, len(index[word])) for word in words]
    # sort by the second element
    word_counts = sorted(word_counts, key = lambda x: int(x[1]))
    # return the list, but only the first element of each set (the tokens)
    return [word for word, _ in word_counts]

def search(index, query):
    """ Return the document ids for documents matching the query. Assume that
    query is a single string, possibly containing multiple words. The steps
    are to:
    1. tokenize the query
    2. Sort the query words by the length of their postings list
    3. Intersect the postings list of each word in the query.

    If a query term is not in the index, then an empty list should be returned.

    Params:
      index...An inverted index (dict mapping words to document ids)
      query...A string that may contain multiple search terms. We assume the
      query is the AND of those terms by default.

    E.g., below we search for documents containing 'a' and 'b':
    >>> search({'a': [0, 1], 'b': [1, 2, 3], 'c': [4]}, 'a b')
    [1]
    """
    tokens = tokenize(query)
    sort = sort_by_num_postings(tokens, index)
    # [1,2,3,4]
    # a = intersection of 1 and 2
    # b = intersection of a and 3
    # c = intersection of b and 4
    token_count = len(sort)
    if token_count == 0:
        cur_intersection = []
    elif token_count == 1:
        cur_intersection = index[sort[0]]
    else:
        cur_intersection = intersect(index[sort[0]], index[sort[1]]) 
        for i in sort[2:]:
            cur_intersection = intersect(cur_intersection, index[i]) 
    return cur_intersection

def main():
    """ Main method. You should not modify this. """
    documents = open('documents.txt').readlines()
    tokens = [tokenize(d) for d in documents]
    index = create_index(tokens)
    queries = open('queries.txt').readlines()
    for query in queries:
        results = search(index, query)
        print('\n\nQUERY:%s\nRESULTS:\n%s' % (query, '\n'.join(documents[r] for r in results)))


if __name__ == '__main__':
    main()
