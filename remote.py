#########################
# define all the remote operation based 'ssh' command
#########################

import os, subprocess, sys
import collections


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def cmd(cmd_str):
    s = subprocess.check_output(cmd_str, shell=True)
    return s.strip().split('\n')


def ssh_find(ssh, find_dir, find_name, other_cmd=''):
    return cmd('ssh {} "find {} -name {} {}"'.format(ssh, find_dir, find_name, other_cmd))


def ssh_find_modify_time(host, find_dir, find_name, time_tag='Modify'):
    """
    get the file modify time
    Args:
        host: ssh host
        find_dir: host dir
        find_name: file name
        time_tag: Modify / Access / Change

    Returns:
        a dict whose keys are file name and values are time string
    """
    cmd_out = ssh_find(host, find_dir, find_name, other_cmd='-exec stat {} \;')
    
    res = collections.OrderedDict()
    for line in cmd_out:
        a = line.split(None, 1)
        if a[0].startswith('File:'):
            file_name = a[1].strip("'")
        elif a[0].startswith(time_tag + ':'):
            file_time = a[1]
            res[file_name] = file_time

    return res


def ssh_find_the_latest_file(host, find_dir, find_name, time_tag='Modify'):
    """
    Returns:
        file name, file time
    """
    ftimes = ssh_find_modify_time(host, find_dir, find_name, time_tag)
    ftimes = sorted(ftimes.items(), key=lambda x: x[1], reverse=True)
    return ftimes[0]


def download(host, remote_path, local_path):
    os.system('rsync -av {}:{} {}'.format(host, remote_path, local_path))


def download_ckpt(host, ckpt_dir, ckpt_name='*.ckpt', local_dir='.'):

    mkdirs(local_dir)
    
    latest_ckpt, _ = ssh_find_the_latest_file(host, ckpt_dir, ckpt_name + '.index')
    print('download the latest ckpt: %s' % latest_ckpt)
    os.system('rsync -av {}:{} {}'.format(host, latest_ckpt[0:-len('index')] + '*', local_dir))


def download_ckpt_based_checkpoint(host, checkpoint_path, local_dir='.'):
    mkdirs(local_dir)

    s = cmd('ssh {} cat {}'.format(host, checkpoint_path))[0]
    ckpt_path = s.split(None, 1)[1].strip('"\n')

    ckpt_dir = os.path.dirname(checkpoint_path)
    ckpt_name = os.path.split(ckpt_path)[-1]

    try:
        download_ckpt(host, ckpt_dir, ckpt_name, local_dir)
    except subprocess.CalledProcessError:
        print('cannot find the ckpt: %s/%s' % (ckpt_dir, ckpt_name))










