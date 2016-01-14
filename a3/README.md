## Assignment 3: Classification

In this assignment, you'll implement a Multinomial Naive Bayes classifier for spam filtering.

Complete the classify.py method by following Figure 13.2 from your text (using add-one smoothing). I recommend reading Example 13.1 to test your understanding of this computation.

The main method downloads the data [here](http://cs.iit.edu/~culotta/cs429/lingspam.zip), which is a slightly modified version of the [LingSpam](http://www.aueb.gr/users/ion/docs/ir_memory_based_antispam_filtering.pdf) spam dataset. This contains a set of emails categorized as spam or not, with headers removed.

The output of running `python classify.py` should match that in [Log.txt](Log.txt).

**Please note:** This assignment asks for a Multinomial, not Bernoulli, Naive Bayes. See book and notes for distinction.
