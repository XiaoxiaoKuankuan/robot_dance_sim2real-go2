import time
import os
import csv
import lcm
import numpy as np
import torch
# import cv2

from go2_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
TEST = False

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

        # 定义保存路径
        save_dir = 'data'  
        self.file_path = os.path.join(save_dir, 'robot_data.csv')

        # 如果目录不存在，创建目录
        os.makedirs(save_dir, exist_ok=True)

        self.se = se  # State Estimator（状态估计器）

        self.dt = 0.02
        self.timestep = 0

        self.num_obs = 42
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
        self.default_dof_pos = [0.1, 0.8, -1.5,
                                -0.1, 0.8, -1.5,
                                0.1, 0.8, -1.5,
                                -0.1, 0.8, -1.5]  # LF RF LH RH  0.8

        # 改成go2的
        self.p_gains = 20
        self.d_gains = 0.5

        self.actions = torch.zeros(12)
        # self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)

        self.joint_idxs = self.se.joint_idxs  # RF LF RH LH
        self.reset()

        # 测试动作参数
        self._targetPos_1 = np.array([[0.0, 1.36, -2.65], [0.0, 1.36, -2.65],
                                      [-0.2, 1.36, -2.65], [0.2, 1.36, -2.65]]).flatten()

        self._targetPos_2 = np.array([[0.0, 0.67, -1.3], [0.0, 0.67, -1.3],
                                      [0.0, 0.67, -1.3], [0.0, 0.67, -1.3]]).flatten()

        self._targetPos_3 = np.array([[-0.35, 1.36, -2.65], [0.35, 1.36, -2.65],
                                      [-0.5, 1.36, -2.65], [0.5, 1.36, -2.65]]).flatten()

        self.startPos = np.zeros(12)
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True

    def get_obs(self):

        # self.body_angular_vel = self.se.get_body_angular_vel()
        self.body_angular_vel = np.array(self.se.get_body_angular_vel())
        print("self.body_angular_vel is :", self.body_angular_vel)
        self.gravity_vector = self.se.get_gravity_vector()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()

        ob = np.concatenate(( self.body_angular_vel.reshape(1, -1) * self.scales["ang_vel"],
                                self.gravity_vector.reshape(1, -1),
                             (self.dof_pos - self.default_dof_pos).reshape(1, -1) * self.scales["dof_pos"],
                             self.dof_vel.reshape(1, -1) * self.scales["dof_vel"],
                             self.actions.detach().numpy().reshape(1, -1)  # 确保 actions 是 NumPy
                             ), axis=1)
        # 裁剪观察，限制在 self.scales["clip_observations"] 范围内
        ob = np.clip(ob, -self.scales["clip_observations"], self.scales["clip_observations"])
        # print("dof_pos is :", self.dof_pos)

        return torch.tensor(ob, device=self.device).float()

    def publish_action(self):

        command_for_robot = pd_tau_targets_lcmt()

        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = self.joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = np.full(12, self.p_gains)
        command_for_robot.kd = np.full(12, self.d_gains)
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        # 计算控制力矩  没有直接用于 command_for_robot，可能是仅用于监测或记录
        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        # 由lcm将神经网络输出的action传入c++ sdk
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def test_action(self):
        if self.firstRun:
            self.startPos = self.se.get_dof_pos()
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            self.actions_scaled = (1 - self.percent_1) * self.startPos + self.percent_1 * self._targetPos_1

        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            self.actions_scaled = (1 - self.percent_2) * self._targetPos_1 + self.percent_2 * self._targetPos_2

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            self.actions_scaled = self._targetPos_2.copy()

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            self.actions_scaled = (1 - self.percent_4) * self._targetPos_2 + self.percent_4 * self._targetPos_3

    def step(self, actions):

        if TEST:
            self.test_action()
            print('actions:', self.actions_scaled)
        print('actions:', actions)  
        clip_actions = self.scales["clip_actions"]/self.scales["action_scale"]
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # self.last_actions = self.actions[:]
        self.actions_scaled= ( self.actions[0, :12].detach().cpu().numpy() * self.scales["action_scale"]).flatten()
        
        self.actions_scaled += self.default_dof_pos  # 偏移量+默认关节角度

        self.joint_pos_target = self.actions_scaled[self.joint_idxs]  # 调整腿顺序
        # print('joint_pos_target:', self.joint_pos_target)
        self.publish_action()  # 由lcm将神经网络输出的action传入c++ sdk
        # time.sleep(max(self.dt - (time.time() - self.time), 0))  # 确保固定的循环频率（50Hz）
        # if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        # self.time = time.time()

        obs = self.get_obs()

        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.concatenate((obs.detach().cpu().numpy().flatten(), self.joint_pos_target.flatten())))

        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],  # 关节位置
            "joint_vel": self.dof_vel[np.newaxis, :],  # 关节速度
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],  # 目标关节位置
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],  # 目标关节速度
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],  # 机身线速度
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],  # 机身角速度
        }

        # self.timestep += 1
        return obs, None, None, infos