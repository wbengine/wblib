#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import collections

import subprocess
from StringIO import StringIO


# compare two word sequence (array), and return the error number
def wer(hypos, refer, special_word=None):
    """
    compute the err number

    For example:

        refer = A B C
        hypos = X B V C

        after alignment

        refer = ~A B ^  C
        hypos = ~X B ^V C
        err number = 2

        where ~ denotes replacement error, ^ denote insertion error, * denotes deletion error.

    if set the special_word (not None), then the special word in reference can match any words in hypothesis.
    For example:
        refer = 'A <?> C'
        hypos = 'A B C D C'

        after aligment:

        refer =  A   <?> C
        hypos =  A B C D C

        where <?> matches 'B C D'. error number = 0

    Usage:
        ```
        refer = 'A <?> C'
        hypos = 'A B C D C'

        res = wer.wer(refer, hypos, '<?>')
        print('err={}'.format(res['err']))
        print('ins={ins} del={del} rep={rep}'.format(**res))
        print('refer = {}'.format(' '.join(res['refer'])))
        print('hypos = {}'.format(' '.join(res['hypos'])))
        ```



    :param refer: a string or a list of words
    :param hypos: a string or a list of hypos
    :param special_word: this word in reference can match any words
    :return:
        a result dict, including:
        res['word']: word number
        res['err']:  error number
        res['del']:  deletion number
        res['ins']:  insertion number
        res['rep']:  replacement number
        res['hypos']: a list of words, hypothesis after alignment
        res['refer']: a list of words, reference after alignment
    """

    res = {'word': 0, 'err': 0, 'none': 0, 'del': 0, 'ins': 0, 'rep': 0, 'hypos': [], 'refer': []}

    refer_words = refer if isinstance(refer, list) else refer.split()
    hypos_words = hypos if isinstance(hypos, list) else hypos.split()

    hypos_words.insert(0, '<s>')
    hypos_words.append('</s>')
    refer_words.insert(0, '<s>')
    refer_words.append('</s>')

    hypos_len = len(hypos_words)
    refer_len = len(refer_words)

    if hypos_len == 0 or refer_len == 0:
        return res

    go_nexts = [[0, 1], [1, 1], [1, 0]]
    score_table = [([['none', 10000, [-1, -1], '', '']] * refer_len) for i in range(hypos_len)]
    score_table[0][0] = ['none', 0, [-1, -1], '', '']  # [error-type, note distance, best previous]

    for i in range(hypos_len - 1):
        for j in range(refer_len):
            for go_nxt in go_nexts:
                nexti = i + go_nxt[0]
                nextj = j + go_nxt[1]
                if nexti >= hypos_len or nextj >= refer_len:
                    continue

                next_score = score_table[i][j][1]
                next_state = 'none'
                next_hypos = ''
                next_refer = ''

                if go_nxt == [0, 1]:
                    next_state = 'del'
                    next_score += 1
                    next_hypos = '*' + ' ' * len(refer_words[nextj])
                    next_refer = '*' + refer_words[nextj]

                elif go_nxt == [1, 0]:
                    next_state = 'ins'
                    next_score += 1
                    next_hypos = '^' + hypos_words[nexti]
                    next_refer = '^' + ' ' * len(hypos_words[nexti])

                else:
                    if special_word is not None and refer_words[nextj] == special_word:
                        for ii in range(i+1, hypos_len-1):
                            next_score += 0  # can match any words, without penalty
                            next_state = 'none'
                            next_refer = special_word
                            next_hypos = ' '.join(hypos_words[i+1:ii+1])

                            if next_score < score_table[ii][nextj][1]:
                                score_table[ii][nextj] = [next_state, next_score, [i, j], next_hypos, next_refer]

                        # avoid add too many times
                        next_score = 10000

                    else:
                        next_hypos = hypos_words[nexti]
                        next_refer = refer_words[nextj]
                        if hypos_words[nexti] != refer_words[nextj]:
                            next_state = 'rep'
                            next_score += 1
                            next_hypos = '~' + next_hypos
                            next_refer = '~' + next_refer

                if next_score < score_table[nexti][nextj][1]:
                    score_table[nexti][nextj] = [next_state, next_score, [i, j], next_hypos, next_refer]

    res['err'] = score_table[hypos_len - 1][refer_len - 1][1]
    res['word'] = refer_len - 2
    i = hypos_len - 1
    j = refer_len - 1
    refer_fmt_words = []
    hypos_fmt_words = []
    while i >= 0 and j >= 0:
        res[score_table[i][j][0]] += 1  # add the del/rep/ins error number
        hypos_fmt_words.append(score_table[i][j][3])
        refer_fmt_words.append(score_table[i][j][4])
        [i, j] = score_table[i][j][2]

    refer_fmt_words.reverse()
    hypos_fmt_words.reverse()

    # format the hypos and refer
    assert len(refer_fmt_words) == len(hypos_fmt_words)
    for i in range(len(refer_fmt_words)):
        w = max(len(refer_fmt_words[i]), len(hypos_fmt_words[i]))
        fmt = '{:>%d}' % w
        refer_fmt_words[i] = fmt.format(refer_fmt_words[i])
        hypos_fmt_words[i] = fmt.format(hypos_fmt_words[i])

    res['refer'] = refer_fmt_words[0:-1]
    res['hypos'] = hypos_fmt_words[0:-1]

    return res


def read_to_dict(f, type=str):
    d = collections.OrderedDict()
    with open(f) as fin:
        for line in fin:
            a = line.split()
            d[a[0]] = type(' '.join(a[1:]))

    return d


def file_wer(refer_file, hypos_file, special_word=None, log_file=None):
    """
    comput the WER on best file

    :param refer_file:
    :param hypos_file:
    :param special_word: this word in reference can match any words
    :param log_file:
    :return:
    """

    nLine = 0
    nTotalWord = 0
    nTotalErr = 0

    if log_file is None:
        log_io = sys.stdout
    else:
        log_io = open(log_file, 'wt')

    # read to dict
    refer_dict = read_to_dict(refer_file)
    hypos_dict = read_to_dict(hypos_file)

    for key, hypos in hypos_dict.items():
        res = wer(refer_dict[key], hypos, special_word)
        nTotalErr += res['err']
        nTotalWord += res['word']

        log_io.write('[{}] {}\n'.format(nLine, key))
        log_io.write('[err={err}] {err}/{word} {total_err}/{total_word} ins={ins} del={del} rep={rep}\n'.format(
            total_err=nTotalErr, total_word=nTotalWord, **res))
        log_io.write('refer: ' + ' '.join(res['refer']) + '\n')
        log_io.write('hypos: ' + ' '.join(res['hypos']) + '\n')
        log_io.flush()

        nLine += 1

    if log_file is not None:
        log_io.close()

    return [nTotalErr, nTotalWord, 1.0 * nTotalErr / nTotalWord * 100]
