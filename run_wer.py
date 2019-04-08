#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import base


def main():
    parser = argparse.ArgumentParser(
        usage='python run_wer.py --best a.best --refer a.refer --filter wer_filter --special_word <?> > wer.log'
    )
    parser.add_argument('--best', help='kaldi 格式的1-best文件，每行包括 label sentence')
    parser.add_argument('--refer', help='kaldi 格式的refer文件，格式与best相同')
    parser.add_argument('--filter', help='a filter file used in sed, filter被作用于best和refer', default=None)
    parser.add_argument('--special_word', help='a special word, refer中的 special word 可以用来匹配任意多的词', default='<?>')
    args = parser.parse_args()

    best_file = args.best
    refer_file = args.refer

    assert base.exists(best_file), 'cannot find best file %s' % best_file
    assert base.exists(refer_file), 'cannot find refer file %s' % refer_file

    if args.filter is not None:
        print('run sed filter ...')
        base.filter_text(best_file, best_file + '.filter', args.filter)
        base.filter_text(refer_file, refer_file + '.filter', args.filter)

        best_file += '.filter'
        refer_file += '.filter'

    print('compute WER')
    print('best = %s' % best_file)
    print('refer = %s' % refer_file)
    err, word, wer = base.CmpWER(best_file, refer_file, sys.stdout)

    print('\n[Finished]')
    print('best = %s' % args.best)
    print('refer = %s' % args.refer)
    print('filter = %s' % args.filter)
    print('errs = %d' % err)
    print('words = %d' % word)
    print('wer = %.8f' % wer)


if __name__ == '__main__':
    main()
