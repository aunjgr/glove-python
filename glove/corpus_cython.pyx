#!python
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
import scipy.sparse as sp

from libc.stdlib cimport malloc, free

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map


cdef inline int int_min(int a, int b) nogil: return a if a <= b else b


cdef int words_to_ids(list words, vector[int]& word_ids,
                      dictionary, int supplied, int ignore_missing):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied, a word is missing from it,
    and we are not ignoring out-of-vocabulary (OOV) words, an
    error value of -1 is returned.

    If we have an OOV word and we do want to ignore them, we use
    a -1 placeholder for it in the word_ids vector to preserve
    correct context windows (otherwise words that are far apart
    with the full vocabulary could become close together with a
    filtered vocabulary).
    """

    cdef int word_id

    word_ids.resize(0)

    if supplied == 1:
        for word in words:
            # Raise an error if the word
            # is missing from the supplied
            # dictionary.
            word_id = dictionary.get(word, -1)
            if word_id == -1 and ignore_missing == 0:
                return -1

            word_ids.push_back(word_id)

    elif supplied == -1:
        for word in words:
            word_ids.push_back(word)

    else:
        for word in words:
            word_id = dictionary.setdefault(word,
                                            len(dictionary))
            word_ids.push_back(word_id)

    return 0


def construct_cooccurrence_matrix(corpus, dictionary, int supplied,
                                  int window_size, int ignore_missing):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # Declare the cooccurrence map
    cdef unordered_map[unsigned long, float] matrix

    # String processing variables.
    cdef list words
    cdef int i, j, outer_word, inner_word
    cdef int wordslen, window_stop, error
    cdef vector[int] word_ids

    # Pre-allocate some reasonable size
    # for the word ids vector.
    word_ids.reserve(1000)

    # Iterate over the corpus.
    for words in corpus:

        # Convert words to a numeric vector.
        error = words_to_ids(words, word_ids, dictionary,
                             supplied, ignore_missing)
        if error == -1:
            raise KeyError('Word missing from dictionary')

        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            # Continue if we have an OOD token.
            if outer_word == -1:
                continue

            if window_size >= 0:
                window_stop = int_min(i + window_size + 1, wordslen)

                for j in range(i + 1, window_stop):
                    inner_word = word_ids[j]

                    if inner_word == -1:
                        continue

                    # Do nothing if the words are the same.
                    if inner_word == outer_word:
                        continue

                    if inner_word < outer_word:
                        matrix[(<unsigned long>inner_word << 32) + outer_word] += 1.0 / (j - i)
                    else:
                        matrix[(<unsigned long>outer_word << 32) + inner_word] += 1.0 / (j - i)
            else:
                for j in range(i + 1, wordslen):
                    inner_word = word_ids[j]

                    if inner_word == -1:
                        continue

                    if inner_word < outer_word:
                        matrix[(<unsigned long>inner_word << 32) + outer_word] += 1.0
                    else:
                        matrix[(<unsigned long>outer_word << 32) + inner_word] += 1.0

    # Create the matrix.
    row_np = np.empty(len(matrix), dtype=np.int32)
    col_np = np.empty(len(matrix), dtype=np.int32)
    data_np = np.empty(len(matrix), dtype=np.float64)

    j = 0

    for item in matrix:
        row_np[j] = item.first >> 32
        col_np[j] = item.first & <unsigned int> 0xffffffff
        data_np[j] = item.second

        j += 1

    # Create and return the matrix.
    dim = col_np.max() + 1
    return sp.coo_matrix((data_np, (row_np, col_np)), shape=(dim, dim), dtype=np.float64)
