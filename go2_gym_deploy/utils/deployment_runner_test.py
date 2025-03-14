import copy
import time
import os

import numpy as np
import torch

from go2_gym_deploy.utils.logger import MultiLogger


class DeploymentRunner:
    def __init__(self, experiment_name="unnamed", se=None, log_root="."):
        self.agents = {}
        self.policy = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.button_states = np.zeros(4)

        self.is_currently_probing = False  # 标记当前是否在进行“探测”过程。探测通常是指测试环境、设备或状态的过程
        self.is_currently_logging = [False, False, False, False]  # 表示四个按钮是否正在进行日志记录

    def init_log_filename(self):
        # 初始化日志文件路径并创建目录
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")

        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue

    # 将一个控制代理添加到 agents 字典，并指定控制代理的名称
    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)
    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        self.policy = policy

    # 通过校准过程，使机器人处于一个稳定的、已知的起始姿态
    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        # control_obs = None  # 预先定义，避免未赋值错误
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,])
                else:
                    final_goal = np.zeros(12)
                nominal_joint_pos = agent.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    self.button_states = self.se.get_buttons()
                    if self.se.state_estimator.right_lower_right_switch_pressed:
                        print(">>>>>>>>>>>>>>> R2 is pressed <<<<<<<<<<<<<")
                        self.se.state_estimator.right_lower_right_switch_pressed = False
                        break

                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = joint_pos - nominal_joint_pos
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    next_target = target

                    cal_action[:, 0:12] = next_target
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    self.button_states = self.se.get_buttons()
                    if self.se.state_estimator.right_lower_right_switch_pressed:
                        print(">>>>>>>>>>>>>>> R2 is pressed again <<<<<<<<<<<<<")
                        self.se.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs

