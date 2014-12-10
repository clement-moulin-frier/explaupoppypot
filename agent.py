from numpy import zeros, ones, array

from explauto.utils.config import make_configuration
from explauto.models.dmp import DmpPrimitive
from explauto.utils import bounds_min_max
from explauto.agent import Agent


class DmpAgent(Agent):
    def __init__(self, n_dmps, n_bfs, used, default, conf, sm, im, dmp_type='discrete', ay=None):
        Agent.__init__(self, conf, sm, im)
        self.n_dmps, self.n_bfs = n_dmps, n_bfs
        self.current_m = zeros(self.conf.m_ndims)
        if ay is None:
            self.dmp = DmpPrimitive(n_dmps, n_bfs, used, default, type=dmp_type)
        else:
            self.dmp = DmpPrimitive(n_dmps, n_bfs, used, default,
                                    type=dmp_type, ay=ones(n_dmps) * 1.)

    @classmethod
    def from_settings(cls, starting_position, n_bfs, sm, babbling, im,
                      bfs_half_range, angle_half_range, s_mins, s_maxs):
        n_dmps = len(starting_position)

        default = zeros(n_dmps*(n_bfs+2))
        default[:n_dmps] = starting_position
        default[-n_dmps:] = starting_position

        poppy_ag = {'m_mins': list([-bfs_half_range] * (n_dmps * n_bfs)) +
                    list(default[:n_dmps] - angle_half_range),

                    'm_maxs': list([bfs_half_range] * (n_dmps * n_bfs)) +
                    list(default[:n_dmps] + angle_half_range),

                    's_mins': s_mins,
                    's_maxs': s_maxs
                    }

        ag_conf = make_configuration(**poppy_ag)

        sm_model = sm['cls'](ag_conf, **sm['config'])

        im_dims = ag_conf.m_dims if babbling == 'motor' else ag_conf.s_dims
        im_model = im['cls'](ag_conf, im_dims, **im['config'])

        used = array([False]*n_dmps + [True]*(n_dmps*n_bfs) + [True]*n_dmps)

        return cls(n_dmps=n_dmps, n_bfs=n_bfs, used=used, default=default,
                   conf=ag_conf, sm=sm_model, im=im_model)

    def motor_primitive(self, m):
        self.m = bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        y = self.dmp.trajectory(self.m)
        self.current_m = y[-1, :]
        return y  # y[:int(len(y) * ((n_bfs*2. - 1.)/(n_bfs*2.))), :]

    def sensory_primitive(self, s):
        return s[-1]  # array([mean(s)]) #s[[-1]]
