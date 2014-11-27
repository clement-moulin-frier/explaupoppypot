import pickle

from numpy import array, zeros
from numpy.random import randn

from explauto.sensorimotor_model.nearest_neighbor import NearestNeighbor
from explauto.utils.config import make_configuration
from explauto.experiment import Experiment
from explauto.utils import rand_bounds
from explauto import InterestModel

from pyvrep.xp import PoppyVrepXp
from pyvrep.pool import VrepXpPool

from environment import VrepEnvironment, scene, conf, poppy_config
from agent import DmpAgent




sms = {
    'knn': (NearestNeighbor, {'sigma_ratio': 1. / 28}),
}

# motors = ['l_hip_x', 'l_hip_z', 'l_hip_y', 'l_knee_y', 'l_ankle_y']

# eval_at = [2, 10]
# tc = load('tc-25.npy')[:3]

gui = True


class PoppyXp(PoppyVrepXp):
    def __init__(self, babbling_name, im_name, sm_name, iter):

        self.babbling_name, self.im_name, self.sm_name = babbling_name, im_name, sm_name
        self.tag = 'xp-{}_{}-{}-{}.pickle'.format(im_name, babbling_name, sm_name, iter)
        PoppyVrepXp.__init__(self, scene, gui=gui)

    def bootstrap(self, expe, n, bootstap_range_div=28.):

        conf = make_configuration(expe.ag.conf.m_mins/bootstap_range_div, expe.ag.conf.m_maxs / bootstap_range_div, expe.ag.conf.s_mins/bootstap_range_div, expe.ag.conf.s_maxs/bootstap_range_div)

        m_rand = rand_bounds(conf.m_bounds, n=n)
        for m in m_rand:
            m[-expe.ag.dmp.n_dmps:] = expe.ag.default[:expe.ag.dmp.n_dmps] + conf.m_ranges[-expe.ag.dmp.n_dmps:] * randn(expe.ag.dmp.n_dmps)
            mov = expe.ag.motor_primitive(m)
            s = expe.env.update(mov, log=False)
            s = expe.ag.sensory_primitive(s)
            expe.ag.sensorimotor_model.update(m)

    def run(self):
        print 'run'
        env = VrepEnvironment(self.robot, **conf)
        n_dmps = env.conf.m_ndims
        n_bfs = 2

        default = zeros(n_dmps*(n_bfs+2))
        angle_limits = []
        for i, m in enumerate(env.robot.motors):
            default[i] = m.present_position
            default[i + (n_dmps*(n_bfs+1))] = m.present_position
            angle_limits.append(poppy_config['motors'][m.name]['angle_limit'])
        default = array(default)
        angle_limits = array(angle_limits)
        motor_ranges = angle_limits[:, 1] - angle_limits[:, 0]

        poppy_ag = {'m_mins': list([-100] * (n_dmps * n_bfs)) + list(angle_limits[:, 0]), # + [-40.] * n_dmps, # + list(angle_limits[:,0]),
                    'm_maxs': list([100] * (n_dmps * n_bfs)) + list(angle_limits[:, 1]), # + [40.] * n_dmps, # + list(angle_limits[:,1]),
                    's_mins': [-0.5, -0.7, 0. ],
                    's_maxs': [1., 0.7, 0.7]
                   }

        poppy_ag_conf = make_configuration(**poppy_ag)

        im_dims = poppy_ag_conf.m_dims if self.babbling_name == 'motor' else poppy_ag_conf.s_dims
        im = InterestModel.from_configuration(poppy_ag_conf, im_dims, self.im_name)

        sm_cls, kwargs = sms[self.sm_name]
        sm = sm_cls(poppy_ag_conf, **kwargs)

        used = array([False]*n_dmps + [True]*(n_dmps*n_bfs) + [True]*n_dmps)
        ag = DmpAgent(n_dmps, n_bfs, used, default, poppy_ag_conf, sm, im)
        print 'ag', ag.conf.m_ndims

        print 'Running xp', self.tag

        xp = Experiment(env, ag)

        env.unsubscribe('motor', xp)
        env.unsubscribe('sensori', xp)
        ag.subscribe('movement', xp)
        # xp.evaluate_at(eval_at, tc)
        xp.run(10)

        with open('logs/{}'.format(self.tag), 'wb') as f:
            pickle.dump(xp.log, f)


if __name__ == '__main__':
    SM = ('knn', )
    IM = ('motor', 'goal')
    print 'creating xp'
    expes = [PoppyXp('goal', 'random', 'knn', 0), PoppyXp('motor', 'random', 'knn', 0)]
    # expes[0].setup()
    # expes[0].run()
    #expes[0].start()
    pool = VrepXpPool(expes)
    pool.run(2)

    # pool = VrepXpPool([IkXp(im, sm, i + 1)
                       # for i in range(3) for sm in SM for im in IM])
    # pool.run(2)