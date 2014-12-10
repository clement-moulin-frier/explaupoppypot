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
import datetime
import os

if __name__ == '__main__':
	date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = 'logs/' + date
        os.mkdir(log_dir)
	with open('/tmp/expe.pbs', 'w') as f:
		f.write(pbs.format(log_dir=log_dir, job_name=date))
	call(["qsub", "/tmp/expe.pbs"])
