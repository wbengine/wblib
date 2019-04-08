import math
import os
import sys
import time
import io
import numpy as np
import shutil
import json
import platform
import re
import copy
from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy


# get the platform
def is_window():
    return platform.system() == 'Windows'


def is_linux():
    return platform.system() == 'Linux'


def rate2str(a, b):
    # return string a / b
    return '{:.2f}% ({} / {})'.format(100.0 * a / b, a, b)


# from itertools import zip_longest
def exists(path):
    return os.path.exists(path)


# create the dir
def mkdir(path, is_recreate=False, force=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if is_recreate:
            if force:
                b = 'y'
            else:
                b = input('Path exit: {} \n'
                          'Do you want delete it? [y|n]: '.format(path))

            if b == 'y' or b == 'yes':
                print('Delete and recreate path', path)
                rmdir(path)
                os.makedirs(path)
    return path


# get the file name
def path_file_name(path, rm_suffix=False):
    s = os.path.split(path)[-1]
    if rm_suffix:
        s = os.path.splitext(s)[0]
    return s


def mklogdir(path, logname='trf.log', is_recreate=False, force=False):
    mkdir(path, is_recreate, force)
    sys.stdout = std_log(os.path.join(path, logname))
    return path


# prepare the log dir
def prepare_log_dir(logdir, logname):
    # create the logdir
    mkdir(logdir, is_recreate=True)
    # output to log
    sys.stdout = std_log(os.path.join(logdir, logname))
    # print
    print('[{}] log to {}'.format(__name__, os.path.join(logdir, logname)))


# remove files
def remove(path):
    if os.path.exists(path):
        os.remove(path)


# remove a dir
def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# make sure a folder
def folder(path):
    if path[-1] != os.sep:
        return path + os.sep
    return path


# get the name of current script
def script_name():
    argv0_list = sys.argv[0].split(os.sep)
    name = argv0_list[len(argv0_list) - 1]
    name = name[0:-3]
    return name


def file_avoid_overwrite(path):
    """if the path is exit, then revise the file to avoid overwritting"""
    if os.path.exists(path):
        t = time.localtime()
        data_code = '{:04d}{:02d}{:02d}[{:02d}{:02d}]'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        idx = path.rfind('.')
        if idx == -1:
            path += '_' + data_code
        else:
            path = path[0:idx] + '_' + data_code + path[idx:]
    return path


def log_head(contents=''):
    """return a string, including the date and time"""
    s = '================================================================\n'
    s += '[time]   ' + time.asctime(time.localtime()) + '\n'
    s += '[name]   ' + script_name() + '\n'
    s += '[author] Bin Wang\n'
    s += '[email]  wb.th08@gmail.com\n'
    s += contents
    s += '===============================================================\n'
    return s


def log_tail(contents=''):
    """return a string, which can be write to the tail of a log files"""
    s = '================================================================\n'
    s += '[time]   ' + time.asctime(time.localtime()) + '\n'
    s += '[program finish succeed!]\n'
    s += contents
    s += '===============================================================\n'
    return s


@contextmanager
def processing(s):
    print('[wblib.processing] ' + s)
    beg_time = time.time()
    yield
    print('[wblib.processing] finished! time={:.2}m'.format((time.time()-beg_time)/60))


def pprint_dict(d, is_print=True):
    prep_d = {}
    for k, v in d.items():
        new_v = v
        if isinstance(new_v, np.int32) or isinstance(new_v, np.int64):
            new_v = int(new_v)
        if isinstance(new_v, np.ndarray):
            new_v = list(new_v)
        if isinstance(new_v, list):
            if len(new_v) > 10:
                new_v = '[' + ', '.join([str(i) for i in new_v[0: 8]] + ['...', str(new_v[-1])]) + ']'
            else:
                new_v = json.dumps(new_v)
        if isinstance(new_v, tuple):
            new_v = json.dumps(new_v)
        if isinstance(new_v, dict):
            new_v = pprint_dict(new_v, is_print=False)
        prep_d[k] = new_v

    if is_print:
        print(json.dumps(prep_d, sort_keys=True, indent=4))
    return prep_d


def json_formulate(d):
    if isinstance(d, dict):
        iter_list = d.items()
    elif isinstance(d, list):
        iter_list = list(enumerate(d))
    elif isinstance(d, tuple):
        iter_list = list(enumerate(d))
    else:
        raise TypeError('unsuport the type {}'.format(type(d)))

    res_list = []
    for k, v in iter_list:
        if isinstance(v, np.int32) or isinstance(v, np.int64):
            v = int(v)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            v = float(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, dict) or isinstance(v, list) or isinstance(v, tuple):
            v = json_formulate(v)
        else:
            try:
                json.dumps(v)
            except TypeError:
                v = str(v)

        res_list.append((k, v))

    if isinstance(d, dict):
        return dict(res_list)
    elif isinstance(d, list):
        return [v for k, v in res_list]
    elif isinstance(d, tuple):
        return tuple([v for k, v in res_list])
    else:
        raise TypeError('unsuport the type {}'.format(type(d)))



def interpolate(a, b, weight):
    """output:
        if a==None: return b
        else: return a*weight + b*(1-weight)
    """
    if a is None:
        return b
    else:
        return weight * a + (1 - weight) * b


class clock:
    def __init__(self):
        self.time_beginning = time.time()
        self.time_recoder = dict()
        self.time_recoder_beg_stack = list()

        self.print_recorde_info = False

    def beg(self):
        self.time_beginning = time.time()

    def end(self, unit='m'):
        """
        return the record time
        :param unit: one of 's'(second) 'm' (minute) 'h' (hours)
        :return:
        """
        if unit == 's':
            k = 1
        elif unit == 'm':
            k = 60
        elif unit == 'h':
            k = 3600
        else:
            raise KeyError('[end] unknown time unit = ' + unit)

        res = (time.time() - self.time_beginning)/k
        self.beg()
        return res

    @contextmanager
    def recode(self, name):
        try:
            self.time_recoder.setdefault(name, 0)
            self.time_recoder_beg_stack.append(time.time())
            if self.print_recorde_info:
                print('[clock] recode [{}] beg'.format(name))
            yield
        finally:
            beg_time = self.time_recoder_beg_stack.pop(-1)
            self.time_recoder[name] += (time.time() - beg_time)/60
            if self.print_recorde_info:
                print('[clock] recode [{}] end, time={}'.format(name, self.time_recoder[name]))

    def items(self):
        return sorted(self.time_recoder.items(), key=lambda x: x[0])

    def merge(self, clock_variables):
        # add the time_recoder in the given clock into the current clock
        self.time_recoder.update(clock_variables.time_recoder)


class std_log:
    """output text to log file and console"""
    def __init__(self, log_file=None, mod='at'):
        """
        init
        :param log_file: the log file, if None, then write to the <script_name>.log
        :param mod: 'wt' or 'at' (default)
        """
        self.__console__ = sys.__stdout__
        if log_file is None:
            log_file = script_name() + '.log'
        self.__log__ = open(log_file, mod)

    def __del__(self):
        self.__log__.close()

    def write(self, output_stream):
        self.__console__.write(output_stream)
        self.__log__.write(output_stream)
        self.__log__.flush()

    def flush(self):
        self.__console__.flush()
        self.__log__.flush()


# count the sentence number and the word number of a txt files
def file_count(fname):
    if isinstance(fname, str):
        f = open(fname)
        nLine = 0
        nWord = 0
        for line in f:
            nLine += 1
            nWord += len(line.split())
        f.close()
    else:
        """input is a list of list"""
        nLine = len(fname)
        nWord = sum([len(x) for x in fname])
    return [nLine, nWord]


def file_len_count(fname):
    """
    count the file length
    """
    len_dict = {}
    with open(fname, 'rt') as f:
        for line in f:
            n = len(line.split())
            len_dict.setdefault(n, 0)
            len_dict[n] += 1

    max_len = max(len_dict.keys())
    len_count = np.zeros(max_len + 1)
    for n, count in len_dict.items():
        len_count[n] = count

    return len_count.tolist()

# get more info for txt files
class TxtInfo(object):
    def __init__(self, fname):
        """
        Count the information of input txt files
        Args:
            fname: str/ or a list of list
        """
        self.nLine = 0
        self.nWord = 0
        self.vocab = {}
        self.min_len = 100
        self.max_len = 0
        self.fname = fname

        if isinstance(fname, str):
            with open(fname) as f:
                for line in f:
                    a = line.split()
                    self.nLine += 1
                    self.nWord += len(a)
                    self.min_len = min(self.min_len, len(a))
                    self.max_len = max(self.max_len, len(a))
                    for w in a:
                        self.vocab.setdefault(w, 0)
                        self.vocab[w] += 1
        else:
            for a in fname:
                self.nLine += 1
                self.nWord += len(a)
                self.min_len = min(self.min_len, len(a))
                self.max_len = max(self.max_len, len(a))
                for w in a:
                    self.vocab.setdefault(w, 0)
                    self.vocab[w] += 1
        self.nVocab = len(self.vocab)

    def __str__(self):
        return 'line={:,} word={:,} vocab={:,} minlen={} maxlen={}'.format(
            self.nLine, self.nWord, self.nVocab, self.min_len, self.max_len)

    def write(self, fp):
        if isinstance(self.fname, str):
            fp.write(self.fname + '\n')
        fp.write('line={:,}\nword={:,}\nvocab={:,}\nminlen={}\nmaxlen={}\n'.format(
            self.nLine, self.nWord, self.nVocab, self.min_len, self.max_len))
        print(str(self))


# rmove the frist column of each line
def file_rmlabel(fread, fout):
    with open(fread) as f1, open(fout, 'wt') as f2:
        for line in f1:
            f2.write(' '.join(line.split()[1:]) + '\n')


# get the word list in files
def getLext(fname):
    v = dict()
    f = open(fname)
    for line in f:
        words = line.split()
        for w in words:
            w = w.upper()  # note: to upper
            n = v.setdefault(w, 0)
            v[w] = n + 1
    f.close()

    # resorted
    n = 0
    for k in sorted(v.keys()):
        v[k] = n
        n += 1
    return v


# corpus word to number
# the id of a word w  = v[w] + id_offset (if w in v) or v[unk]+ id_offset (if w not in v)
def corpus_w2n(fin, fout, v, unk='<UNK>', id_offset=0):
    f = open(fin)
    fo = open(fout, 'wt')
    for line in f:
        words = line.split()
        nums = []
        for w in words:
            w = w.upper()
            if w in v:
                nums.append(v[w])
            elif unk in v:
                nums.append(v[unk])
            else:
                print('[wb.corpus_w2n]: cannot find the key = ' + w);
                exit()

            fo.write(''.join(['{} '.format(n + id_offset) for n in nums]))

        f.close()
        fo.close()


# corpus

# ppl to loglikelihood
# two usage: PPL2LL(ppl, nline, nword), PPL2LL(ppl, file)
def PPL2LL(ppl, obj1, obj2=0):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return -math.log(ppl) * (nLine + nWord) / nLine


# LL to PPL
# two usage: LL2PPL(LL, nline, nword), LL2PPL(LL, file)
def LL2PPL(LL, obj1, obj2=0):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return np.exp(-LL * nLine / (nLine + nWord))


# LL incrence bits to PPL decence precent
def LLInc2PPL(LLInc, obj1, obj2):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return 1 - np.exp(-LLInc * nLine / (nLine + nWord))


# read nbest file into dict
def nbest_read_to_dict(nbest_file, type=str):
    d = OrderedDict()
    with open(nbest_file) as f:
        for line in f:
            a = line.split()
            label = '-'.join(a[0].split('-')[0:-1])
            val = type(' '.join(a[1:]))

            d.setdefault(label, [])
            d[label].append(val)

    return d


# compare two word sequence (array), and return the error number
def TxtScore(hypos, refer, special_word=None):
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
    score_table[0][0] = ['none', 0, [-1, -1], '<s>', '<s>']  # [error-type, note distance, best previous]

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

    res['refer'] = refer_fmt_words[1:-1]
    res['hypos'] = hypos_fmt_words[1:-1]

    return res



# calculate the WER given best file
def CmpWER(best, temp, log_str_or_io=None, sentence_process_fun=None, special_word=None):
    nLine = 0
    nTotalWord = 0
    nTotalErr = 0

    f1 = open(best) if isinstance(best, str) else best
    f2 = open(temp) if isinstance(temp, str) else temp
    fout = open(log_str_or_io, 'wt') if isinstance(log_str_or_io, str) else log_str_or_io

    # using the label to match sentences
    # first load the correct sentences
    temp_dict = dict()
    for line in f2:
        a = line.split()
        temp_dict[a[0]] = a[1:]

    # process each sentence in the 1best file
    for line in f1:
        a = line.split()
        try:
            temp_sent = temp_dict[a[0]]
        except KeyError:
            raise("[{}.CmpWER] cannot find the label={} in template".format(__name__, a[0]))

        target = a[1:]
        if sentence_process_fun is not None:
            target = sentence_process_fun(target)
            temp_sent = sentence_process_fun(temp_sent)

        res = TxtScore(target, temp_sent, special_word=special_word)
        nTotalErr += res['err']
        nTotalWord += res['word']

        if fout is not None:
            fout.write('[{}] {}\n'.format(nLine, a[0]))
            fout.write('[nDist={0}] [{0}/{1}] [{2}/{3}]\n'.format(res['err'], res['word'], nTotalErr, nTotalWord))
            fout.write('refer: ' + ''.join([i + ' ' for i in res['refer']]) + '\n')
            fout.write('hypos: ' + ''.join([i + ' ' for i in res['hypos']]) + '\n')
            fout.flush()

        nLine += 1

    if isinstance(best, str):
        f1.close()
    if isinstance(temp, str):
        f2.close()
    if isinstance(log_str_or_io, str):
        fout.close()

    return [nTotalErr, nTotalWord, 1.0 * nTotalErr / nTotalWord * 100]


def CmpCER(best, temp, log_str_or_io=None):
    def word_to_chars(wseq):
        char_seq = []
        for w in wseq:
            #  split Chinese word to char, and preserve the English words
            cs = re.split(r'([\u4e00-\u9fa5])', w)
            cs = list(filter(None, cs))
            char_seq += cs
        return char_seq
    return CmpWER(best, temp, log_str_or_io, sentence_process_fun=word_to_chars)


def TxtScoreForPool(input_tuple):
    # input_tuple = (label, target, temp_sent)
    return input_tuple[0], TxtScore(input_tuple[1], input_tuple[2])


def CmpOracleWER(nbest, refer, log_str_or_io=None, sentence_process_fun=None, processes=4):
    import tqdm
    from multiprocessing import Pool

    f1 = open(nbest) if isinstance(nbest, str) else nbest
    f2 = open(refer) if isinstance(refer, str) else refer
    fout = open(log_str_or_io, 'wt') if isinstance(log_str_or_io, str) else log_str_or_io

    # using the label to match sentences
    # first load the correct sentences
    temp_dict = OrderedDict()
    for line in f2:
        a = line.split()
        temp_dict[a[0]] = a[1:]

    def iter_get_data():
        for line in f1:
            a = line.split()
            head = a[0]

            idx = head.rindex('-')
            label = head[0:idx]
            num = int(head[idx + 1:])

            target = a[1:]
            temp_sent = temp_dict[label]

            if sentence_process_fun is not None:
                target = sentence_process_fun(target)
                temp_sent = sentence_process_fun(temp_sent)

            yield label, target, temp_sent

    total_lines = len(f1.readlines())
    f1.seek(0)

    # perform wer
    nbest_dct = OrderedDict()
    pool = Pool(processes=processes)
    for label, res in tqdm.tqdm(pool.imap(func=TxtScoreForPool, iterable=iter_get_data()),
                                desc='computing oracle WER',
                                total=total_lines):
        dist = nbest_dct.setdefault(label, [])
        dist.append(res)

    # compute the oracle wer
    lowest_err = []
    for label, err_dict_list in nbest_dct.items():
        fout.write(label + ' ')
        for res in err_dict_list:
            fout.write('{}/{}/{}/{}/{} '.format(res['ins'], res['del'], res['rep'], res['err'], res['word']))
        fout.write('\n')

        err_list = [(res['err'], res['word']) for res in err_dict_list]
        lowest_err.append(sorted(err_list, key=lambda x: x[0])[0])

    a = np.array(lowest_err).sum(axis=0)
    nTotalErr = a[0]
    nTotalWord = a[1]

    if isinstance(nbest, str):
        f1.close()
    if isinstance(refer, str):
        f2.close()
    if isinstance(log_str_or_io, str):
        fout.close()

    return nTotalErr, nTotalWord, 1.0 * nTotalErr / nTotalWord * 100


# given the score get the 1-best result
def GetBest(nbest, score, best):
    """
    :param nbest: nbest file
    :param score: score file, if None, then select the first sentence
    :param best: output the best file
    """

    nbest_dict = nbest_read_to_dict(nbest)

    if score is None:
        score_dict = None
    elif isinstance(score, str) and os.path.isfile(score):
        score_dict = nbest_read_to_dict(score)
    else:
        # a list of scores
        score_dict = OrderedDict()
        i = 0
        for label, vals in nbest_dict.items():
            score_dict[label] = score[i: i+len(vals)]
            i += len(vals)

    # fout
    fout = open(best, 'wt') if isinstance(best, str) else best

    for label, vals in nbest_dict.items():
        if score_dict is not None:
            s = score_dict[label]
            i = np.argmin(s)
        else:
            i = 0
        fout.write('{} {}\n'.format(label, vals[i]))

    if isinstance(best, str):
        fout.close()


# load the score file
def LoadScore(fname):
    s = []
    f = open(fname, 'rt')
    for line in f:
        a = line.split()
        s.append(float(a[1]))
    f.close()
    return np.array(s)


# Load the nbest/score label
def LoadLabel(fname):
    s = []
    f = open(fname, 'rt')
    for line in f:
        a = line.split()
        s.append(a[0])
    f.close()
    return s


# Write Score
def WriteScore(fname, s, label=[]):
    with open(fname, 'wt') as f:
        for i in range(len(s)):
            if len(label) == 0:
                f.write('line={}\t{}\n'.format(i, s[i]))
            else:
                f.write('{}\t{}\n'.format(label[i], s[i]))


# cmp interpolate
def ScoreInterpolate(s1, s2, w):
    s1 = LoadScore(s1) if isinstance(s1, str) else s1
    s2 = LoadScore(s2) if isinstance(s2, str) else s2
    return w * s1 + (1-w) * s2


# tune the lmscale and acscale to get the best WER
def TuneWER(nbest, temp, lmscore, acscore, lmscale, acscale=[1]):
    opt_wer = 100
    opt_lmscale = 0
    opt_acscale = 0
    if isinstance(lmscore, str):
        lmscore = LoadScore(lmscore)
    if isinstance(acscore, str):
        acscore = LoadScore(acscore)
    # tune the lmscale
    for ac in acscale:
        for lm in lmscale:
            s = ac * np.array(acscore) + lm * np.array(lmscore)
            # best_file = 'lm{}.ac{}.best'.format(lmscale, acscale)
            best_file = io.StringIO()
            GetBest(nbest, s, best_file)
            best_file.seek(0)

            [totale, totalw, wer] = CmpWER(best_file, temp)

            # print('acscale={}\tlmscale={}\twer={}\n'.format(acscale, lmscale, wer))
            if wer < opt_wer:
                opt_wer = wer
                opt_lmscale = lm
                opt_acscale = ac

            # remove the best files
            # os.remove(best_file)
            best_file.close()

    return opt_wer, opt_lmscale, opt_acscale


# Res file, such as
# model LL-train LL-valid LL-test PPL-train PPL-valid PPL-test
# Kn5  100 100 100 200 200 200
# rnn 100 100 100  200 200 200
class FRes:
    def __init__(self, fname, print_to_cmd=False):
        self.fname = fname  # the file name
        self.data = [] # recore all the data in files
        self.head = [] # recore all the label
        self.comment = []  # comments
        self.print_to_cmd = print_to_cmd
        self.new_add_name = ''  # record the current add name

    # load data from file
    def Read(self):
        self.data = []
        self.head = []
        self.comment = ''
        if os.path.exists(self.fname):
            with open(self.fname, 'rt') as f:
                nline = 0
                for line in f:
                    a = line.split()
                    if len(a) == 0:
                        continue

                    if a[0][0] == '#':
                        # comments if the first character of the first word is #
                        self.comment += line
                    else:
                        if nline == 0:
                            self.head = a
                        else:
                            self.data.append(a)
                        nline += 1
        else:
            self.head.append('models')

        # return all the name in files
        names = []
        for a in self.data:
            names.append(a[0])
        return names

    # write data to file
    def Write(self):
        n = len(self.head)
        width = [len(i) for i in self.head]
        for a in self.data:
            for i in range(len(a)):
                width[i] = max(width[i], len(a[i]))

        with open(self.fname, 'wt') as f:
            # write comments
            f.write(self.comment + '\n')
            # write data
            for a in [self.head] + self.data:
                outputline = ''
                for i in range(len(a)):
                    outputline += '{0:{width}}'.format(a[i], width=width[i]+2)
                f.write(outputline + '\n')

                # print the new added line
                if self.print_to_cmd and a[0] == self.new_add_name:
                    print(outputline)

    # add a line comment
    def AddComment(self, s):
        self.Read()
        self.comment += '# ' + s + '\n'
        self.Write()

    # clean files
    def Clean(self):
        remove(self.fname)

    # remove default head
    def RMDefaultHead(self):
        self.head = ['models']


    # get the head
    def GetHead(self):
        self.Read()
        return self.head


    # get ['KN', '100', '111', 1213']
    def GetLine(self, name):
        self.Read()
        for a in self.data:
            if a[0] == name:
                return a
        print('[FRes] Cannot find {}'.format(name))
        return []

    def GetValue(self, name, tag):
        """
        get the value in files
        :param name: the line name
        :param tag:  the column name, str/int, if int, then get the 'tag'-th line
        :return: the values
        """
        if isinstance(tag, int):
            column = tag
        else:
            head = self.GetHead()
            column = head.index(tag)
        line = self.Get(name)
        return line[column]

    # get ['KN', 100, 111, 1213]
    def Get(self, name):
        a = self.GetLine(name)
        res = [a[0]]
        for w in a[1:]:
            res.append(float(w))
        return res
    # add a line
    def AddLine(self, name):
        self.Read()
        for a in self.data:
            if a[0] == name:
                return a
        self.data.append([name])
        return self.data[-1]
    # add datas, such as Add('KN5', ['LL-train', 'LL-valid'], [100, 10] )
    def Add(self, name, head, value):
        name = name.replace(' ', '_')
        a = self.AddLine(name)
        for w in head:
            if w not in self.head:
                self.head.append(w)
            i = self.head.index(w)
            
            if len(a) < len(self.head):
                a.extend(['0']* (len(self.head) - len(a)))
            v = value[ head.index(w) ]
            if isinstance(v, str):
                a[i] = v
            elif isinstance(v, float):
                a[i] = '{:.3f}'.format(float(v))
            else:
                a[i] = '{}'.format(v)
        self.new_add_name = name
        self.Write()

    def AddWER(self, name, wer):
        self.Add(name, ['WER'], [wer])

    def AddLLPPL(self, name, LL, PPL):
        self.Add(name, [a+'-'+b for a in ['LL','PPL'] for b in ['train','valid','test']], LL+PPL)
        
    def AddLL(self, name, LL, txt):
        PPL = [0] * len(LL)
        for i in range(len(LL)):
            [sents, words] = file_count(txt[i])
            PPL[i] = LL2PPL(-LL[i], sents, words)
        self.Add(name, [a+'-'+b for a in ['LL','PPL'] for b in ['train','valid','test']], LL+PPL)

    def AddPPL(self, name, PPL, txt):
        """
        Add the PPL and NLL to the RES files
        :param name: name of model
        :param PPL:  list, the ppls on training/valid/test set
        :param txt:  list, the txt files, or the list of list
        :return: None
        """
        LL = [0] * len(PPL)
        for i in range(len(PPL)):
            [sents, words] = file_count(txt[i])
            LL[i] = -PPL2LL(PPL[i], sents, words)
        self.Add(name, [a+'-'+b for a in ['LL','PPL'] for b in ['train','valid','test']], LL+PPL)


def log_load_column(file_name, column=None, to_type=str, line_head=None):
    """
    read the log information form log files,
    each line is like 'step=0 epoch=0.1 time=1.2 NLL=100'
    :param file_name: file name or file_handle returned by open()
    :param column: a list of string/integer, denoting the label of each line (such as 'step','epoch') or the column number
    :param to_type: transform the data to 'to_type'
    :param line_head: if not None, just process the line whose head is line_head
    :return a dict
    """
    if isinstance(file_name, str):
        f = open(file_name)
    else:
        f = file_name

    res = dict()
    for line in f:
        if line_head is not None and line.find(line_head) != 0:
            continue

        a = line.split()
        labels = []
        values = []
        undef_num = 0
        for s in a:
            v = s.split('=')
            if len(v) == 1:
                labels.append('<undef-%d>' % undef_num)
                values.append(v[0])
                undef_num += 1
            else:
                labels.append(v[0])
                values.append(v[1])

        # add to res_dict
        for lab, v in zip(labels, values):
            if column is None or lab in column:
                if lab not in res:
                    res[lab] = []

                try:
                    trans_v = to_type(v)
                except ValueError:
                    trans_v = v
                res[lab].append(trans_v)

    if isinstance(file_name, str):
        f.close()

    return res


class FileBank(object):
    """
    A set of files can be used to dynamically open and automatically closed
    """
    def __init__(self, default_file_name=None):
        self.file_dict = dict()
        self.default_file_name = default_file_name

    def __del__(self):
        for key, file in self.file_dict.items():
            file.close()

    def get(self, name, default_file_name=None, default_file_mod='wt'):
        if name not in self.file_dict:
            if default_file_name is None:
                if self.default_file_name is None:
                    raise TypeError('please input a valid file name!')
                default_file_name = self.default_file_name + '.' + name
            self.file_dict[name] = open(default_file_name, default_file_mod)
        return self.file_dict[name]


class ArrayUpdate:
    """
    Update methods support Adam, Gradient Descent.
    """
    def __init__(self, params, opt_config):
        """
        Args:
            params: int or np.array, the params or the size of the params
            opt_config: such as {'name': 'adam', 'max_norm': 10, 'clip': 10}
        """
        self.opt_config = dict(opt_config)
        self.opt_name = self.opt_config.setdefault('name', 'adam')

        self.t = 0
        self.param_size = params if isinstance(params, int) else np.shape(params)
        if self.opt_name.lower() == 'adam':
            self.m = np.zeros(self.param_size)
            self.v = np.zeros(self.param_size)

        if self.opt_name.lower() == 'amsgrad':
            self.m = np.zeros(self.param_size)
            self.v = np.zeros(self.param_size)
            self.v_max = np.zeros(self.param_size)

        self.dx_norm = 0

    def prepare_grad(self, steps):
        if 'max_norm' in self.opt_config:
            max_norm = self.opt_config['max_norm']
            if max_norm > 0:
                self.dx_norm = np.sqrt(np.sum(steps**2))
                if self.dx_norm > max_norm:
                    steps *= max_norm / self.dx_norm

        if 'clip' in self.opt_config:
            clip = self.opt_config['clip']
            steps = np.clip(steps, a_min=-clip, a_max=clip)

        return steps

    def update(self, grads, lr):
        grads = np.array(grads)
        if self.opt_name.lower() == 'adam':
            return self.update_adam(grads, lr)
        elif self.opt_name.lower() == 'amsgrad':
            return self.update_amsgrad(grads, lr)
        else:
            return self.update_grad(grads, lr)

    def update_grad(self, grads, lr):
        # gradient descent
        # update method: params += update_grad
        return -lr * self.prepare_grad(grads)

    def update_adam(self, grads, lr):
        # adam
        beta1 = 0.9
        beta2 = 0.999
        es = 10**-8
        self.t += 1
        self.m = beta1 * self.m + (1-beta1) * grads
        self.v = beta2 * self.v + (1-beta2) * (grads**2)
        m_tide = self.m / (1-beta1**self.t)
        v_tied = self.v / (1-beta2**self.t)
        step = m_tide / (np.sqrt(v_tied) + es)

        return -lr * self.prepare_grad(step)

    def update_amsgrad(self, grads, lr):
        beta1 = 0.9
        beta2 = 0.999
        es = 10 ** -8
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grads
        self.v = beta2 * self.v + (1 - beta2) * (grads ** 2)
        self.v_max = np.maximum(self.v_max, self.v)

        step = self.m / (np.sqrt(self.v_max) + es)
        return -lr * self.prepare_grad(step)


class PPrintObj(object):
    def pprint(self):
        pprint_dict(self.get_print_dict())

    def get_print_dict(self):
        d = dict()
        for key, v in self.__dict__.items():
            if isinstance(v, PPrintObj):
                new_v = v.get_print_dict()
            else:
                new_v = self.value2str(v)
            d[key] = new_v
        return d

    def value2str(self, v):
        return v


def json_load(fp):
    chunk = ''
    for line in fp:
        chunk += line
        try:
            return json.loads(chunk)
        except ValueError:
            pass
    return None


class Config(object):
    value_encoding_map = {
        np.ndarray: lambda x: x.tolist(),
        np.int16: int,
        np.int32: int,
        np.int64: int,
        np.float32: float,
        np.float16: float
    }

    def compact_list(self, v, max_size=10):
        if isinstance(v, list):
            if len(v) > max_size:
                s_list = [str(i) for i in v[0:5]] + ['...'] + [str(i) for i in v[-2:]]
                return '[' + ', '.join(s_list) + ']'
        return v

    def encode(self, is_compact=False):
        d = dict()
        for key, v in self.__dict__.items():
            if isinstance(v, Config):
                d[key] = v.encode()
            else:
                trans_fun = lambda x: x
                for t, f in Config.value_encoding_map.items():
                    if isinstance(v, t):
                        trans_fun = f
                d[key] = trans_fun(v) if not is_compact else self.compact_list(trans_fun(v))
        return d

    def decode(self, d):
        for key, v in d.items():
            if isinstance(v, dict):
                c = Config()
                c.decode(v)
                self.__dict__[key] = c
            else:
                self.__dict__[key] = v

    def __str__(self):
        return json.dumps(self.encode(), indent=4, sort_keys=True)

    def printf(self):
        print(json.dumps(self.encode(is_compact=True), indent=4, sort_keys=True))

    def loads(self, s):
        self.decode(json.loads(s))

    def save(self, fp):
        json.dump(self.encode(), fp, indent=4, sort_keys=True)
        fp.write('\n')

    def restore(self, fp):
        self.decode(json_load(fp))

    @staticmethod
    def load(fp):
        c = Config()
        c.restore(fp)
        return c

    def update(self, sub_config):
        """
        update the values based the input config
        Args:
            sub_config: a config, can be the sub-class of the current class

        Returns:
            None
        """
        self.__dict__.update(sub_config.__dict__)


def write_array(f, a, fmt='%+15.5e', delimiter=' ', newline='\n'):
    """write array a to a 'wt' opened file f """
    if a.ndim == 1:
        a.tofile(f, delimiter, fmt)
        f.write(newline)
    elif a.ndim == 2:
        for b in a:
            write_array(f, b, fmt, delimiter, newline)
    elif a.ndim == 3:
        for i, b in enumerate(a):
            write_array(f, b, fmt, delimiter, newline)
            if i < len(a)-1:
                f.write(newline)
    else:
        for b in a:
            write_array(f, b, fmt, delimiter, newline)


class Operation(object):
    def __init__(self):
        self.perform_next_epoch = 0
        self.perform_per_epoch = 1.0

    def run(self, step, epoch):
        if epoch >= self.perform_next_epoch:
            self.perform(step, epoch)

            self.perform_next_epoch += self.perform_per_epoch
            while self.perform_next_epoch < epoch:
                self.perform_next_epoch += self.perform_per_epoch

    def perform(self, step, epoch):
        pass


def logaddexp(a, b, wa=None, wb=None):
    """
    compute log(wa * exp(a) + wb * exp(b))
    Returns:

    """
    if wa is None:
        wa = 1
    if wb is None:
        wb = 1
    m = np.maximum(a, b)
    return np.log(wa * np.exp(a-m) + wb * np.exp(b-m)) + m


def split_to_char_ch(s, keep_en_words=True):
    """
    input a chinese string, and return the char list
    """
    if keep_en_words:
        # split Chinese word to char, and preserve the English words
        cs = re.split(r'([\u4e00-\u9fa5])', s)
        cs = list(filter(None, cs))
    else:
        cs = list(s)
    return cs


def generate_pos(word_len):
    if word_len == 0:
        return []
    if word_len == 1:
        return ['s']
    return ['b'] + ['m'] * (word_len-2) + ['e']


def read_to_dict(txt_file, filter_fun=None, value_fun=str):
    data = OrderedDict()
    with open(txt_file) as f:
        for line in f:
            a = line.strip().split(None, 1)
            label = a[0]
            if len(a) >= 2:
                info = a[1]
            else:
                info = ''

            if filter_fun is not None and not filter_fun(label):
                continue

            if label in data:
                # create a new label
                i = 0
                new_label = label
                while new_label in data:
                    i += 1
                    new_label = label + '-%d' % i
                label = new_label
            data[label] = value_fun(info)
    return data

def write_to_txt(txt_file, dict_or_list, write_label=True, file_format='wt'):
    if isinstance(dict_or_list, dict) or isinstance(dict_or_list, OrderedDict):
        vals = dict_or_list.items()
    else:
        vals = dict_or_list

    with open(txt_file, file_format) as f:
        for label, info in vals:
            if write_label:
                f.write('{} {}\n'.format(label, info))
            else:
                f.write('{}\n'.format(info))


import subprocess
def filter_text(input, output, sed_filter_file=None):
    input_lines = []
    with open(input) as f:
        for line in f:
            input_lines.append('  ' + '  '.join(line.split()) + '  ')

    sed_filter_file = sed_filter_file if sed_filter_file else os.path.join(os.path.dirname(__file__), 'filters/wer_hyp_filter')
    p = subprocess.Popen('sed -f {}'.format(sed_filter_file), stdin=subprocess.PIPE, stdout=open(output, 'wt'), shell=True)
    p.communicate('\n'.join(input_lines))


def audio_duration(path):
    _, s = subprocess.Popen('sox {} -n stat'.format(path), stderr=subprocess.PIPE, shell=True).communicate()
    
    for line in s.strip().split('\n'):
        if line.lower().startswith('length (seconds):'):
            return float(line.split()[-1])
    raise TypeError('sox ERROR: \n{}'.format(s))


def generate_wp(wp_tools_dir, text_list_or_file, bpe_codes, vocabulary=None, output_file=None, to_lower=False):
    """
    Args:
        wp_tools_dir:
        text_list_or_file:
        bpe_codes:
        vocabulary:
        output_file: if None, then return a txt list
        to_lower: if True, then convert text to lower

    Returns:

    """
    cmd_str = os.path.join(wp_tools_dir, 'apply_bpe.py')
    cmd_str += ' -c ' + bpe_codes
    if vocabulary is not None:
        cmd_str += ' --vocabulary ' + vocabulary

    if to_lower:
        cmd_str = "tr '[:upper:]' '[:lower:]' | " + cmd_str

    if output_file is None:
        stdout = subprocess.PIPE
    else:
        stdout = open(output_file, 'wt')

    if isinstance(text_list_or_file, str):
        # a file
        p = subprocess.Popen(cmd_str, stdin=open(text_list_or_file, 'rt'), stdout=stdout, shell=True)
        pout, _ = p.communicate()
    else:
        # a text list
        p = subprocess.Popen(cmd_str, stdin=subprocess.PIPE, stdout=stdout, shell=True)
        pout, _ = p.communicate('\n'.join(text_list_or_file))

    if pout:
        return pout.decode().strip().split('\n')
    else:
        return ""
