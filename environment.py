import os
import json
import time
from numpy import array

from pypot.primitive import LoopPrimitive

from explauto.environment.environment import Environment
from explauto.exceptions import ExplautoEnvironmentUpdateError

import pypot
import poppytools

configfile = os.path.join(os.path.dirname(poppytools.__file__),
                          'configuration', 'poppy_config.json')

with open(configfile) as f:
    poppy_config = json.load(f)

scene = os.path.join(os.path.dirname(pypot.__file__), '..',
                     'samples', 'notebooks', 'poppy-lying_sticky.ttt')

alias = 'motors'  # 'l_leg'

conf = {
    'motors': alias,
    'move_duration': 4.,
    't_reset': 2.,
    'm_mins': array([-180.] * len(poppy_config['motors'])),
    'm_maxs': array([180.] * len(poppy_config['motors'])),
    's_mins': [-0.5, -0.7, 0.],
    's_maxs': [1., 0.7, 0.7]
}


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
            # raise ExplautoEnvironmentUpdateError("Vrep time < 0")

        if self.elapsed_time > self.primitive_duration:
            self.stop(wait=False)
            return

        i = int(self.elapsed_time / self.one_step_duration)

        if i < 0:
            i = 0
            print "Vrep time < 0"

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

    def compute_sensori_effect(self, m_env):
        raise NotImplementedError("shouldn't use this one")
