import os
import pickle
import datetime

from numpy import array, nan
from numpy.random import randn

# from explauto.sensorimotor_model.nearest_neighbor import NearestNeighbor
from explauto.utils.config import make_configuration
from explauto.experiment import Experiment
from explauto.utils import rand_bounds
# from explauto import InterestModel

from pyvrep.xp import PoppyVrepXp

from environment import VrepEnvironment, scene, conf
from agent import DmpAgent, get_params




# sms = {
#     'knn': (NearestNeighbor, {'sigma_ratio': 1. / 28}),
# }

# motors = ['l_hip_x', 'l_hip_z', 'l_hip_y', 'l_knee_y', 'l_ankle_y']

# eval_at = [2, 10]
# tc = load('tc-25.npy')[:3]

gui = False


class PoppyXp(PoppyVrepXp):
    def __init__(self, babbling_name, im_name, sm_name, n_bfs=2, iter=0):

        self.babbling_name, self.im_name, self.sm_name = babbling_name, im_name, sm_name
        self.n_bfs = n_bfs
        self.tag = 'xp-{}_{}-{}-{}.pickle'.format(im_name, babbling_name, sm_name, iter)
        PoppyVrepXp.__init__(self, scene, gui=gui)


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
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = 'logs/' + date
        os.mkdir(log_dir)

        env = VrepEnvironment(self.robot, **conf)

        ag = DmpAgent(**get_params(self.n_bfs, env.rest_position, self.babbling_name, self.sm_name, self.im_name))

        print 'Running xp', self.tag

        xp = Experiment(env, ag)

        env.unsubscribe('motor', xp)
        env.unsubscribe('sensori', xp)
        ag.subscribe('movement', xp)
        # xp.evaluate_at(eval_at, tc)

        xp.log.env_conf = conf
        xp.log.ag_conf = {'n_bfs': self.n_bfs,
                          'starting_position': env.rest_position,
                          'babbling_name': self.babbling_name,
                          'sm_name': self.sm_name,
                          'im_name': self.im_name
                          }
        xp.log.bootstrap_conf = {'n': 16, 'bootstap_range_div': 28.}
        self.bootstrap(xp, **xp.log.bootstrap_conf)

        log_each = 10
        for run in range(100 / log_each):
            xp.run(log_each)
            with open(log_dir + '/{}'.format(self.tag), 'wb') as f:
                pickle.dump(xp.log, f)
            f.close()
            print 'saved ' + str((run + 1) * log_each)


if __name__ == '__main__':
    # SM = ('knn', )
    # IM = ('motor', 'goal')
    # print 'creating xp'
    expes = [PoppyXp('goal', 'discretized_progress', 'knn', n_bfs=3, iter=0)]
    # expes[0].setup()
    # expes[0].run()
    expes[0].start()
    # pool = VrepXpPool(expes)
    # pool.run(2)

    # pool = VrepXpPool([IkXp(im, sm, i + 1)
                       # for i in range(3) for sm in SM for im in IM])
    # pool.run(2)
