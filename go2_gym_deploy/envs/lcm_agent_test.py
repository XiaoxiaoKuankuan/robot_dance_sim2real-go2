import time

import lcm
import numpy as np
import torch
# import cv2

from go2_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, se):

        self.se = se  # State Estimator（状态估计器）

        self.dt = 0.02
        self.timestep = 0

        self.num_obs = 60
        self.num_actions = 12
        self.num_envs = 1
        self.device = 'cpu'
        # obs_scales还没改

        self.scales = {"lin_vel": 2.0,
                      "ang_vel": 0.25,
                      "dof_pos": 1.0,
                      "dof_vel": 0.05,
                      "height_measurements": 5.0,
                      "clip_observations": 100.,
                      "clip_actions": 2.5,
                      "action_scale": 0.25}
        self.default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1., -1.5, -0.1, 1., -1.5, 0, 0, 0, 0, 0, 0, 0,
                           0]  # LF RF LH RH
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        # 改成go2的
        self.p_gains = [150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 20., 15.,
                   10., 10., 10.]
        self.d_gains = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0.1, 0.1, 0.1, 0.1, 0.1]

        print(f"p_gains: {self.p_gains}")

        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)

        self.joint_idxs = self.se.joint_idxs

        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        self.is_currently_probing = False

    def set_probing(self, is_currently_probing):
        # 探测
        self.is_currently_probing = is_currently_probing


    def get_obs(self):

        self.body_angular_vel = self.se.get_body_angular_vel()
        self.gravity_vector = self.se.get_gravity_vector()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()


        ob = np.concatenate(( self.body_angular_vel.reshape(1, -1) * self.scales["ang_vel"],
                                self.gravity_vector.reshape(1, -1),
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.scales["dof_pos"],
                             self.dof_vel.reshape(1, -1) * self.scales["dof_vel"],
                             self.actions
                             ), axis=1)

        return torch.tensor(ob, device=self.device).float()

    def publish_action(self, action, hard_reset=False):

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * self.scales["action_scale"]).flatten()

        # self.joint_pos_target[[0, 3, 6, 9]] *= -1
        self.joint_pos_target = self.joint_pos_target   # 我们的是不是偏移量？用不用再加初始角度
        self.joint_pos_target += self.default_dof_pos  # 偏移量+默认关节角度
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)
        # print(f'cjp {self.joint_pos_target}')

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        # 计算控制力矩  没有直接用于 command_for_robot，可能是仅用于监测或记录
        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        # 由lcm将神经网络输出的action传入c++ sdk
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def step(self, actions, hard_reset=False):
        clip_actions = self.scales["clip_actions"]
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.last_actions = self.actions[:]
        # self.actions_scaled = self.actions * self.scales["action_scale"]

        self.publish_action(self.actions, hard_reset=hard_reset)  # 由lcm将神经网络输出的action传入c++ sdk
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        self.time = time.time()
        obs = self.get_obs()

        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],  # 关节位置
            "joint_vel": self.dof_vel[np.newaxis, :],  # 关节速度
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],  # 目标关节位置
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],  # 目标关节速度
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],  # 机身线速度
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],  # 机身角速度
        }

        self.timestep += 1
        return obs, None, None, infos