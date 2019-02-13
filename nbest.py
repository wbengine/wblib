import os, sys
import numpy as np

import StringIO
from . import wblib as wb


class NBest(object):
    def __init__(self, nbest, trans, acscore=None, lmscore=None, gfscore=None):
        """
        construct a nbest class

        Args:
            nbest: nbet list
            trans: the test transcript ( correct text )
            acscore: acoustic score
            lmscore: language model score
            gfscore: graph score
        """
        self.nbest = nbest
        self.trans = trans
        self.acscore = None
        self.lmscore = None
        self.gfscore = gfscore
        if acscore is not None:
            self.acscore = wb.LoadScore(acscore)
        if lmscore is not None:
            self.lmscore = wb.LoadScore(lmscore)
        if gfscore is not None:
            self.gfscore = wb.LoadScore(gfscore)

        # save the best result
        self.lmscale = 1.0
        self.acscale = 1.0
        self.total_err = 0
        self.total_word = 0
        self.best_1best = None
        self.best_log = None
        self.wer_per_scale = []

        self.nbest_list_id = None

    def process_best_file(self, best_file):
        new_best_file = wb.io.StringIO()
        for line in best_file:
            new_line = ' '.join(filter(lambda w: w.lower() != '<unk>', line.split()))
            new_best_file.write(new_line + '\n')
        best_file.close()
        new_best_file.seek(0)
        return new_best_file

    def wer(self, lmscale=np.linspace(0.1, 1.0, 10), rm_unk=False, sentence_process_fun=None):
        """
        compute the WER
        Returns:
            word error rate (WER)
        """
        if self.lmscore is None:
            self.lmscore = np.zeros_like(self.acscore)
        if self.acscore is None:
            self.acscore = np.zeros(len(self.lmscore))
        if self.gfscore is None:
            self.gfscore = np.zeros(len(self.lmscore))


        self.wer_per_scale = []

        # tune the lmscale
        opt_wer = 1000
        for ac in [1]:
            for lm in lmscale:
                s = ac * np.array(self.acscore) + lm * (np.array(self.lmscore) + np.array(self.gfscore))
                best_file = StringIO.StringIO()
                log_file = StringIO.StringIO()
                wb.GetBest(self.nbest, s, best_file)
                best_file.seek(0)

                if rm_unk:
                    best_file = self.process_best_file(best_file)

                [totale, totalw, wer] = wb.CmpWER(best_file, self.trans,
                                                  log_str_or_io=log_file,
                                                  sentence_process_fun=sentence_process_fun)

                self.wer_per_scale.append([ac, lm, wer])
                # print('acscale={}\tlmscale={}\twer={}\n'.format(acscale, lmscale, wer))
                if wer < opt_wer:
                    opt_wer = wer
                    self.lmscale = lm
                    self.acscale = ac
                    self.total_word = totalw
                    self.total_err = totale

                    if self.best_1best is not None:
                        self.best_1best.close()
                    self.best_1best = best_file
                    self.best_1best.seek(0)

                    if self.best_log is not None:
                        self.best_log.close()
                    self.best_log = log_file
                    self.best_log.seek(0)

                else:
                    best_file.close()
                    log_file.close()

        return opt_wer

    def oracle_wer(self, rm_unk=False, sentence_process_fun=None):
        import StringIO
        self.oracle_log_io = StringIO.StringIO()

        if rm_unk:
            nbest = self.process_best_file(self.nbest)
        else:
            nbest = self.nbest

        res = wb.CmpOracleWER(nbest, self.trans, self.oracle_log_io, sentence_process_fun)
        self.oracle_log_io.seek(0)
        return res

    def get_trans_txt(self, fwrite):
        # get the transcript text used to calculate PPL
        wb.file_rmlabel(self.trans, fwrite)

    def get_nbest_list(self, data):
        # return the nbest list id files used to rescoring
        if self.nbest_list_id is None:
            self.nbest_list_id = data.load_data(self.nbest, is_nbest=True)

            # # process the empty sequences
            # empty_len = int(data.beg_token_str is not None) + int(data.end_token_str is not None)
            # for s in self.nbest_list_id:
            #     if len(s) == empty_len:
            #         s.insert(-1, data.get_end_token())
        return self.nbest_list_id

    def write_nbest_list(self, fwrite, data):
        with open(fwrite, 'wt') as f:
            for s in self.get_nbest_list(data):
                f.write(' '.join([str(i) for i in s]) + '\n')

    def write_lmscore(self, fwrite):
        with open(fwrite, 'wt') as fout, open(self.nbest, 'rt') as fin:
            for s, line in zip(self.lmscore, fin):
                fout.write(line.split()[0] + '\t' + str(s) + '\n')

    def write_log(self, fname):
        if self.best_log is None:
            print('[{0}.{1}] best_log=None, run {1}.wer() first.'.format(__name__, self.__class__.__name__))
        with open(fname, 'wt') as f:
            self.best_log.seek(0)
            f.write(self.best_log.read())

    def write_1best(self, fname):
        if self.best_1best is None:
            print('[{0}.{1}] best_1best=None, run {1}.wer() first.'.format(__name__, self.__class__.__name__))
        with open(fname, 'wt') as f:
            self.best_1best.seek(0)
            f.write(self.best_1best.read())

    def get_nbest_lens(self):
        return [len(s.split()) - 1 for s in open(self.nbest).readlines()]


