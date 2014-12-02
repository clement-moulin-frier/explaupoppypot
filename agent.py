from numpy import zeros, ones, array

from explauto.sensorimotor_model.nearest_neighbor import NearestNeighbor
from explauto.utils.config import make_configuration
from explauto.models.dmp import DmpPrimitive
from explauto.utils import bounds_min_max
from explauto import InterestModel
from explauto.agent import Agent


class DmpAgent(Agent):
    def __init__(self, n_dmps, n_bfs, used, default, conf, sm, im):
        Agent.__init__(self, conf, sm, im)
        self.n_dmps, self.n_bfs = n_dmps, n_bfs
        self.current_m = zeros(self.conf.m_ndims)
        self.dmp = DmpPrimitive(n_dmps, n_bfs, used, default,
                                type='rythmic', ay=ones(n_dmps) * 1.)

    def motor_primitive(self, m):
        self.m = bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        y = self.dmp.trajectory(self.m)
        self.current_m = y[-1, :]
        return y  # y[:int(len(y) * ((n_bfs*2. - 1.)/(n_bfs*2.))), :]

    def sensory_primitive(self, s):
        return s[-1]  # array([mean(s)]) #s[[-1]]


sms = {
    'knn': (NearestNeighbor, {'sigma_ratio': 1. / 28}),
}

def get_params(babbling_name, sm_name, im_name, env):
    n_dmps = env.conf.m_ndims
    n_bfs = 2

    default = zeros(n_dmps*(n_bfs+2))
    # angle_limits = []
    for i, m in enumerate(env.robot.motors):
        default[i] = m.present_position
        default[i + (n_dmps*(n_bfs+1))] = m.present_position
    default = array(default)

    poppy_ag = {'m_mins': list([-100] * (n_dmps * n_bfs)) + list(default[:n_dmps] - 90.),
                'm_maxs': list([100] * (n_dmps * n_bfs)) + list(default[:n_dmps] + 90.),
                's_mins': [-1., -0.7, -0.1],
                's_maxs': [1., 0.7, 0.7]
                }

    poppy_ag_conf = make_configuration(**poppy_ag)

    im_dims = poppy_ag_conf.m_dims if babbling_name == 'motor' else poppy_ag_conf.s_dims
    im = InterestModel.from_configuration(poppy_ag_conf, im_dims, im_name)

    sm_cls, kwargs = sms[sm_name]
    sm = sm_cls(poppy_ag_conf, **kwargs)

    used = array([False]*n_dmps + [True]*(n_dmps*n_bfs) + [True]*n_dmps)
    return {'n_dmps': n_dmps,
            'n_bfs': n_bfs,
            'used': used,
            'default': default,
            'conf': poppy_ag_conf,
            'sm': sm,
            'im': im
           }
