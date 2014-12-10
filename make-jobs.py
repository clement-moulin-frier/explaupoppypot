pbs = """
#!/bin/sh
#PBS -o {log_dir}/my.output
#PBS -e {log_dir}/my.error
#PBS -l walltime=10:00:0
#PBS -N {job_name}
cd /home/cmoulinf/dev/explaupoppypot
python experiment.py --log_dir {log_dir} --i_expe {i_expe}
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
		i_expe = i + 1
		job_name = os.path.basename(os.path.normpath(args.dir)) + '_' + str(i_expe)
		with open('/tmp/expe.pbs', 'w') as f:
			f.write(pbs.format(log_dir=args.dir, job_name=job_name, i_expe=i_expe))
			call(["qsub", "/tmp/expe.pbs"])
