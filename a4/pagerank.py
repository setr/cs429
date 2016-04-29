""" Assignment 6: PageRank. """
from bs4 import BeautifulSoup, SoupStrainer
from sortedcontainers import SortedList, SortedSet, SortedDict
from collections import Counter
import glob
import os


def compute_pagerank(urls, inlinks, outlinks, b=.85, iters=20):
    """ Return a dictionary mapping each url to its PageRank.
    The formula is R(u) = (1/N)(1-b) + b * (sum_{w in B_u} R(w) / (|F_w|)

    Initialize all scores to 1.0.

    Params:
      urls.......SortedList of urls (names)
      inlinks....SortedDict mapping url to list of in links (backlinks)
      outlinks...Sorteddict mapping url to list of outlinks
    Returns:
      A SortedDict mapping url to its final PageRank value (float)

    >>> urls = SortedList(['a', 'b', 'c'])
    >>> inlinks = SortedDict({'a': ['c'], 'b': set(['a']), 'c': set(['a', 'b'])})
    >>> outlinks = SortedDict({'a': ['b', 'c'], 'b': set(['c']), 'c': set(['a'])})
    >>> sorted(compute_pagerank(urls, inlinks, outlinks, b=.5, iters=0).items())
    [('a', 1.0), ('b', 1.0), ('c', 1.0)]
    >>> iter1 = compute_pagerank(urls, inlinks, outlinks, b=.5, iters=1)
    >>> iter1['a']  # doctest:+ELLIPSIS
    0.6666...
    >>> iter1['b']  # doctest:+ELLIPSIS
    0.333...
    """
    # scores[name] = pagerank score
    scores = SortedDict()
    for u in urls:  # initialize to 1
        scores[u] = 1.

    # left side
    N = len(urls)
    const = (1./N) * (1. - b)

    # right side
    for i in range(iters):
        for u in urls:
            linksum = 0
            for link in inlinks[u]:
                pr = scores[link]
                linksum += pr / len(outlinks[link])
            scores[u] = linksum * b + const
    return scores
            
def get_top_pageranks(inlinks, outlinks, b, n=50, iters=20):
    """
    >>> inlinks = SortedDict({'a': ['c'], 'b': set(['a']), 'c': set(['a', 'b'])})
    >>> outlinks = SortedDict({'a': ['b', 'c'], 'b': set(['c']), 'c': set(['a'])})
    >>> res = get_top_pageranks(inlinks, outlinks, b=.5, n=2, iters=1)
    >>> len(res)
    2
    >>> res[0]  # doctest:+ELLIPSIS
    ('a', 0.6666...
    """
    urls = SortedList(outlinks.keys())
    return sorted(compute_pagerank(urls, inlinks, outlinks, b, iters).items())[:n]
    ###TODO
    pass


def read_names(path):
    """ Do not mofify. Returns a SortedSet of names in the data directory. """
    return SortedSet([os.path.basename(n) for n in glob.glob(path + os.sep + '*')])


def get_links(names, html):
    """
    Return a SortedSet of computer scientist names that are linked from this
    html page. The return set is restricted to those people in the provided
    set of names.  The returned list should contain no duplicates.

    Params:
      names....A SortedSet of computer scientist names, one per filename.
      html.....A string representing one html page.
    Returns:
      A SortedSet of names of linked computer scientists on this html page, restricted to
      elements of the set of provided names.

    >>> get_links({'Gerald_Jay_Sussman'},
    ... '''<a href="/wiki/Gerald_Jay_Sussman">xx</a> and <a href="/wiki/Not_Me">xx</a>''')
    SortedSet(['Gerald_Jay_Sussman'], key=None, load=1000)
    """

    pagenames = SortedSet()
    for link in BeautifulSoup(html, "html.parser", parse_only=SoupStrainer('a')):
        if link.has_attr('href'):
            name = link['href'].split('/')[-1]
            if name in names:
                pagenames.add(name)
    return pagenames
    

def read_links(path):
    """
    Read the html pages in the data folder. Create and return two SortedDicts:
      inlinks: maps from a name to a SortedSet of names that link to it.
      outlinks: maps from a name to a SortedSet of names that it links to.
    For example:
    inlinks['Ada_Lovelace'] = SortedSet(['Charles_Babbage', 'David_Gelernter'], key=None, load=1000)
    outlinks['Ada_Lovelace'] = SortedSet(['Alan_Turing', 'Charles_Babbage'], key=None, load=1000)

    You should use the read_names and get_links function above.

    Params:
      path...the name of the data directory ('data')
    Returns:
      A (inlinks, outlinks) tuple, as defined above (i.e., two SortedDicts)
    """
    inlinks, outlinks = SortedDict(), SortedDict()
    names = read_names(path)

    if os.path.exists(path):  # download and unzip data
        fullpaths = [(name, os.path.join(path, name)) for name in os.listdir(path)]

        for name, f in fullpaths:
            with open(f, 'r') as f:
                html = f.read()
            outlinks[name] = get_links(names, html)
            outlinks[name].discard(name) # no self-links!
            
    for name, links in outlinks.items():
        for l in links:
            try:
                inlinks[l].add(name)
            except KeyError:
                inlinks[l] = SortedSet(name)

    return (inlinks, outlinks)

    # for finding accidental misparses 
    #
    # print ('Checking outlinks')
    # for name in outlinks:
    #     if name not in names:
    #         print ( 'name' )
    # print 
    # print ('Checking inlinks')
    # for name in inlinks:
    #     if name not in names:
    #         print ( 'name' )
    # print
    # print ('Checking names is all found')
    # for name in names:
    #     if name not in inlinks:
    #         print ("missing from inlink:" % name)
    #     if name not in outlinks:
    #         print ("missing from outlink:" % name)

def print_top_pageranks(topn):
    """ Do not modify. Print a list of name/pagerank tuples. """
    print('Top page ranks:\n%s' % ('\n'.join('%s\t%.5f' % (u, v) for u, v in topn)))


def main():
    """ Do not modify. """
    if not os.path.exists('data'):  # download and unzip data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/pagerank.tgz', 'pagerank.tgz')
       tar = tarfile.open('pagerank.tgz')
       tar.extractall()
       tar.close()

    inlinks, outlinks = read_links('data')
    print('read %d people with a total of %d inlinks' % (len(inlinks), sum(len(v) for v in inlinks.values())))
    print('read %d people with a total of %d outlinks' % (len(outlinks), sum(len(v) for v in outlinks.values())))
    topn = get_top_pageranks(inlinks, outlinks, b=.8, n=20, iters=10)
    print_top_pageranks(topn)


if __name__ == '__main__':
    main()
