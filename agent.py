from numpy import zeros, ones
from explauto.models.dmp import DmpPrimitive
from explauto.utils import bounds_min_max
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
