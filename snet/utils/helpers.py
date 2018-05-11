import subprocess


def line_count(filename):
    p = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])
