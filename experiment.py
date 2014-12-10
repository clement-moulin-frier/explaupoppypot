import os
import sys
import pickle
import argparse

from numpy import array, nan
from numpy.random import randn

from explauto.utils.config import make_configuration
from explauto.experiment import Experiment
from explauto.utils import rand_bounds

from pyvrep.xp import VrepXp

from environment import VrepEnvironment
from agent import DmpAgent

gui = False

avakas = 'AVAKAS' in os.environ


class PoppyXp(VrepXp):
    def __init__(self, env_config, ag_config, expe_config,
                 log_dir, i_expe=None, tag='expe_log', description=''):
        self.env_config = env_config
        self.ag_config = ag_config
        self.expe_config = expe_config
        self.log_dir = log_dir
        self.tag = tag
        self.description = description
        self.n_runs = expe_config['n_runs']
        self.log_each = expe_config['log_each']
        self.i_expe = i_expe

        VrepXp.__init__(self, 'poppy', expe_config['scene'])

    def bootstrap(self, expe, n, bootstap_range_div):
        conf = make_configuration(expe.ag.conf.m_centers - expe.ag.conf.m_ranges/(2 * bootstap_range_div),
                                  expe.ag.conf.m_centers + expe.ag.conf.m_ranges/(2 * bootstap_range_div),
                                  expe.ag.conf.s_centers - expe.ag.conf.s_ranges/(2 * bootstap_range_div),
                                  expe.ag.conf.s_centers + expe.ag.conf.s_ranges/(2 * bootstap_range_div))

        m_rand = rand_bounds(conf.m_bounds, n=n)
        for m in m_rand:
            m[-expe.ag.dmp.n_dmps:] = expe.ag.dmp.default[:expe.ag.dmp.n_dmps] + conf.m_ranges[-expe.ag.dmp.n_dmps:] * randn(expe.ag.dmp.n_dmps)
            mov = expe.ag.motor_primitive(m)
            s = expe.env.update(mov, log=True)
            s = expe.ag.sensory_primitive(s)
            expe.ag.sensorimotor_model.update(m, s)
            expe.ag.emit('choice', array([nan] * len(expe.ag.expl_dims)))
            expe.ag.emit('inference', m)
            expe.ag.emit('movement', mov)
            expe.ag.emit('perception', s)
        expe._update_logs()

    def run(self):
        env = VrepEnvironment.from_settings(self.robot, **self.env_config)

        ag = DmpAgent.from_settings(starting_position=env.rest_position,
                                    **self.ag_config)

        print 'Running xp', self.tag

        xp = Experiment(env, ag)

        env.unsubscribe('motor', xp)
        env.unsubscribe('sensori', xp)
        ag.subscribe('movement', xp)

        xp.log.description = self.description

        self.bootstrap(xp, **self.expe_config['bootstrap_config'])

        file = self.log_dir + '/{}'.format(self.tag)
        if self.i_expe is not None:
            file += ('_' + str(self.i_expe))
        file += '.pickle'

        for run in range(self.n_runs / self.log_each):
            xp.run(self.log_each)
            with open(file, 'wb') as f:
                pickle.dump(xp.log, f)
            f.close()
            print 'saved ' + str((run + 1) * self.log_each)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--i_expe', type=int, required=True)
    args = parser.parse_args()
    sys.path.append(args.dir)
    from config import env_config, ag_config, expe_config
    expe = PoppyXp(env_config, ag_config, expe_config,
                   log_dir=args.dir, i_expe=args.i_expe)

    if avakas:
        expe.spawn(avakas=True)
    else:
        expe.spawn(gui=gui)
