pbs = """
#!/bin/sh
#PBS -o {log_dir}/my.output
#PBS -e {log_dir}/my.error
#PBS -l walltime=10:00:0
#PBS -N {job_name}
cd /home/cmoulinf/dev/explaupoppypot
python experiment.py --log_dir {log_dir}
"""

from subprocess import call
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--n_expes', type=int, required=True)
    args = parser.parse_args()
    for i in range(args.n_expes):
        job_name = os.path.basename(os.path.normpath(args.dir)) + '_' + str(i + 1)
        with open('/tmp/expe.pbs', 'w') as f:
            f.write(pbs.format(log_dir=args.dir, job_name=job_name))
            call(["qsub", "/tmp/expe.pbs"])
