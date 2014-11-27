import os
import time
import json
import pickle

from numpy import array, load, zeros, ones

from explauto.sensorimotor_model.nearest_neighbor import NearestNeighbor
from explauto.environment.environment import Environment
from explauto.utils.config import make_configuration
from explauto.models.dmp import DmpPrimitive
from explauto.experiment import Experiment
from explauto.utils import bounds_min_max
from explauto import InterestModel
from explauto.agent import Agent

from pyvrep.xp import PoppyVrepXp
from pyvrep.pool import VrepXpPool

from pypot.primitive import LoopPrimitive

import poppytools
configfile = os.path.join(os.path.dirname(poppytools.__file__),
                                  'configuration', 'poppy_config.json')

with open(configfile) as f:
    poppy_config = json.load(f)

scene = '../../pypot/samples/notebooks/poppy-lying_sticky.ttt'

sms = {
    'knn': (NearestNeighbor, {'sigma_ratio': 1. / 28}),
}

# motors = ['l_hip_x', 'l_hip_z', 'l_hip_y', 'l_knee_y', 'l_ankle_y']
alias = 'motors' #'l_leg'

conf = {
    'motors': alias,
    'move_duration': 4.,
    't_reset': 0.4,
    'm_mins': array([-180.] * len(poppy_config['motors'])),
    'm_maxs': array([180.] * len(poppy_config['motors'])),
    's_mins': [-0.5, -0.7, 0. ],
    's_maxs': [1., 0.7, 0.7]
}

# eval_at = [2, 10]
# tc = load('tc-25.npy')[:3]

class MovementPrimitive(LoopPrimitive):
    def __init__(self, environment, freq, motors, mov, primitive_duration, log):
        LoopPrimitive.__init__(self, environment.robot, freq)
        self.motors = motors
        self.mov = mov
        self.primitive_duration = primitive_duration
        self.one_step_duration = primitive_duration / mov.shape[0]
        self.log = log
        self.env = environment

    def update(self):
        if self.elapsed_time > self.primitive_duration:
            self.stop(wait=False)
            return

        i = int(self.elapsed_time / self.one_step_duration)
        m = self.mov[i, :]
        for i_motor, motor_name in enumerate(self.motors):
            getattr(self.robot, motor_name).goal_position = m[i_motor]
        s = self.env.get_vrep_obj_position('head_visual')
        self.env.s.append(s)
        if self.log:
            self.env.emit('motor', m)
            self.env.emit('sensori', s)


class VrepEnvironment(Environment):
    def __init__(self, robot, motors, move_duration, t_reset,
                 m_mins, m_maxs, s_mins, s_maxs):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.robot = robot
        self.motors = [m.name for m in getattr(self.robot, motors)]
        self.move_duration = move_duration
        self.t_reset = t_reset

    def update(self, m_ag, log=True):
        self.reset()
        head_init = self.get_vrep_obj_position('head_visual')
        self.s = []
        self.motor_primitive = MovementPrimitive(self, 50, self.motors, m_ag, self.move_duration, log)
        self.motor_primitive.start()
        self.motor_primitive.wait_to_stop()
        return array(array(self.s) - array(head_init))

    def get_vrep_obj_position(self, obj):
        io = self.robot._controllers[0].io
        return io.get_object_position(obj)

    def reset(self):
        self.robot.reset_simulation()
        time.sleep(self.t_reset)

    def compute_motor_command(self, m_ag):
        raise NotImplementedError("shouldn't use this one")
    #     m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
    #     return m_env
    #
    def compute_sensori_effect(self, m_env):
        raise NotImplementedError("shouldn't use this one")
    #     pos = dict(zip(self.motors, m_env))
    #     self.robot.goto_position(pos, self.move_duration, wait=True)
    #
    #     time.sleep(.5)
    #
    #     io = self.robot._controllers[0].io
    #     pos = io.get_object_position('foot_left_visual', 'base_link_visual')
    #     # rot = io.get_object_orientation('foot_left_visual', 'base_link_visual')
    #
    #     return pos  # + rot

class DmpAgent(Agent):
    def __init__(self, n_dmps, n_bfs, used, default, conf, sm, im):
        Agent.__init__(self, conf, sm, im)
        self.n_dmps, self.n_bfs = n_dmps, n_bfs
        self.current_m = zeros(self.conf.m_ndims)
        self.dmp = DmpPrimitive(n_dmps, n_bfs, used, default, type='rythmic', ay=ones(n_dmps) * 1.)

    def motor_primitive(self, m):
        self.m = bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        y = self.dmp.trajectory(self.m) # , n_times=2)
        self.current_m = y[-1, :]
        return y #y[:int(len(y) * ((n_bfs*2. - 1.)/(n_bfs*2.))), :] # [::100, :]
    def sensory_primitive(self, s):
         return s[-1] # array([mean(s)]) #s[[-1]]


class PoppyXp(PoppyVrepXp):
    def __init__(self, babbling_name, im_name, sm_name, iter):

        self.babbling_name, self.im_name, self.sm_name = babbling_name, im_name, sm_name
        self.tag = 'xp-{}_{}-{}-{}.pickle'.format(im_name, babbling_name, sm_name, iter)
        PoppyVrepXp.__init__(self, scene, gui=True)

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

        poppy_ag = {
          'm_mins': list([-100] * (n_dmps * n_bfs)) + list(angle_limits[:,0]), # + [-40.] * n_dmps, # + list(angle_limits[:,0]),
          'm_maxs': list([100] * (n_dmps * n_bfs)) + list(angle_limits[:,1]), # + [40.] * n_dmps, # + list(angle_limits[:,1]),
          's_mins': [-0.5, -0.7, 0. ],
          's_maxs': [1., 0.7, 0.7]
        }

        poppy_ag_conf = make_configuration(**poppy_ag)

        im_dims = env.conf.m_dims if self.babbling_name == 'motor' else env.conf.s_dims
        im = InterestModel.from_configuration(env.conf, im_dims, self.im_name)

        sm_cls, kwargs = sms[self.sm_name]
        sm = sm_cls(env.conf, **kwargs)


        used = array([False]*n_dmps + [True]*(n_dmps*n_bfs) + [True]*n_dmps)
        ag = DmpAgent(n_dmps, n_bfs, used, default, poppy_ag_conf, sm, im)

        print 'Running xp', self.tag

        xp = Experiment(env, ag)
        env.unsubscribe('motor', xp)
        env.unsubscribe('sensori', xp)
        ag.subscribe('movement', xp)
        #xp.evaluate_at(eval_at, tc)
        xp.run(10)

        with open('logs/{}'.format(self.tag), 'wb') as f:
            pickle.dump(xp.log, f)


if __name__ == '__main__':
    SM = ('knn', )
    IM = ('motor', 'goal')
    print 'creating xp'
    expes = [PoppyXp('goal', 'random', 'knn', 0)]
    #expes[0].setup()
    #expes[0].run()
    expes[0].start()
    #pool = VrepXpPool(expes)
    #pool.run(2)

    # pool = VrepXpPool([IkXp(im, sm, i + 1)
                       # for i in range(3) for sm in SM for im in IM])
    # pool.run(2)
