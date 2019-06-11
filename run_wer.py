#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import base


usage_str = 'input best file:' \
            '   python run_wer.py --best a.best --refer a.refer --filter wer_filter --special_word <?> > wer.log \n' \
            'input nbest file:' \
            '   python run_wer.py --nbest a.nbest --score a.score --filter wer_filter --special_word <?> > wer.log\n'


def main():

    parser = argparse.ArgumentParser(
        usage=usage_str
    )
    parser.add_argument('--best', help='kaldi 格式的 1-best 文件，每行包括 label sentence')
    parser.add_argument('--nbest', help='kaldi 格式的 n-best file. label={wav_label}-{nbest_id} ')
    parser.add_argument('--score', help='kaldi 格式的 score 文件，与nbest搭配使用，如果为None，则选择nbest中的第一个作为1-best计算WER')
    parser.add_argument('--refer', help='kaldi 格式的 refer 文件，格式与best相同')
    parser.add_argument('--filter', help='a filter file used in sed, filter被作用于best和refer', default=None)
    parser.add_argument('--special_word', help='a special word, refer中的 special word 可以用来匹配任意多的词', default='<?>')
    parser.add_argument('--oracle', action='store_true', help='如果指定，则计算oracle WER')
    args = parser.parse_args()

    best_file = args.best
    nbest_file = args.nbest
    refer_file = args.refer

    assert base.exists(refer_file), 'cannot find refer file %s' % refer_file

    if args.filter is not None:
        print('run sed filter ...')
        if best_file is not None:
            base.filter_text(best_file, best_file + '.filter', args.filter)
            best_file += '.filter'
        if nbest_file is not None:
            base.filter_text(nbest_file, nbest_file + '.filter', args.filter)
            nbest_file += '.filter'

        base.filter_text(refer_file, refer_file + '.filter', args.filter)
        refer_file += '.filter'

    if args.oracle:
        print('compute Oracle WER')
        print('nbest = %s' % nbest_file)
        print('refer = %s' % refer_file)

        err, word, wer = base.CmpOracleWER(nbest_file, refer_file, sys.stdout, special_word=args.special_word)
    else:
        if best_file is None:
            print('nbest to 1-best...')
            best_file = os.path.splitext(args.nbest)[0] + '.best'
            base.GetBest(nbest_file, args.score, best_file)

        print('compute WER')
        print('best = %s' % best_file)
        print('refer = %s' % refer_file)
        err, word, wer = base.CmpWER(best_file, refer_file, sys.stdout, special_word=args.special_word)

    print('\n[Finished]')
    print('best = %s' % args.best)
    print('nbest = %s' % args.nbest)
    print('score = %s' % args.score)
    print('refer = %s' % args.refer)
    print('filter = %s' % args.filter)
    print('errs = %d' % err)
    print('words = %d' % word)
    if args.oracle:
        print('oracle-wer = %.8f' % wer)
    else:
        print('wer = %.8f' % wer)


if __name__ == '__main__':
    main()
