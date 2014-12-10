import os
import json
import time
from numpy import array

from pypot.primitive import LoopPrimitive

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment

import poppytools

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
        self.motors = []
        for i_mot, mot in enumerate(getattr(self.robot, motors)):
            self.motors.append(mot.name)
            # if mot.name in constraints.keys():
            #     m_mins[i_mot] = constraints[mot.name][0]
            #     m_maxs[i_mot] = constraints[mot.name][1]

        time.sleep(4)
        rest_position = []
        for m in self.robot.motors:
            rest_position.append(m.present_position)
        self.rest_position = array(rest_position)

        self.move_duration = move_duration
        self.t_reset = t_reset

    @classmethod
    def from_settings(cls, robot, motors, move_duration, t_reset,
                      m_half_range, s_mins, s_maxs):
        n_motors = len(getattr(robot, motors))
        m_mins = array([-m_half_range] * n_motors)
        m_maxs = array([m_half_range] * n_motors)
        return cls(robot, motors, move_duration, t_reset,
                   m_mins, m_maxs, s_mins, s_maxs)

    def update(self, m_ag, log=True):
        m_env = self.compute_motor_command(m_ag)
        self.reset()
        head_init = self.get_vrep_obj_position('head_visual')
        self.s = []
        self.motor_primitive = MovementPrimitive(self, 50, self.motors, m_env, self.move_duration, log)
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
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m_env):
        raise NotImplementedError("shouldn't use this one")
