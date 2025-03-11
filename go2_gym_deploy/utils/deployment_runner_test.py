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

                    action_scale = 0.25
                    next_target = next_target / action_scale
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


    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs
        # 校准
        control_obs = self.calibrate(wait=True)

        # now, run control loop

        try:
            for i in range(max_steps):

                policy_info = {}
                action = self.policy(control_obs, policy_info)

                for agent_name in self.agents.keys():
                    obs, ret, done, info = self.agents[agent_name].step(action)

                    info.update(policy_info)
                    info.update({"observation": obs, "reward": ret, "done": done, "timestep": i,
                                 "time": i * self.agents[self.control_agent_name].dt, "action": action, "rpy": self.agents[self.control_agent_name].se.get_rpy(), "torques": self.agents[self.control_agent_name].torques})

                    if logging: self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        control_obs, control_ret, control_done, control_info = obs, ret, done, info

                # bad orientation emergency stop
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                # check for logging command
                prev_button_states = self.button_states[:]
                self.button_states = self.se.get_buttons()

                if self.se.left_lower_left_switch_pressed:
                    # 如果按下 left_lower_left_switch 按钮，启动或停止日志记录
                    if not self.is_currently_probing:
                        print("START LOGGING")
                        self.is_currently_probing = True
                        self.agents[self.control_agent_name].set_probing(True)
                        self.init_log_filename()
                        self.logger.reset()
                    else:
                        print("SAVE LOG")
                        self.is_currently_probing = False
                        self.agents[self.control_agent_name].set_probing(False)
                        # calibrate, log, and then resume control
                        control_obs = self.calibrate(wait=False)
                        self.logger.save(self.log_filename)
                        self.init_log_filename()
                        self.logger.reset()
                        time.sleep(1)
                        control_obs = self.agents[self.control_agent_name].reset()
                    self.se.left_lower_left_switch_pressed = False


                if self.se.right_lower_right_switch_pressed:
                    # 检查是否按下 right_lower_right_switch 按钮，如果按下，则进行校准，并等待按钮的再次按下
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.se.right_lower_right_switch_pressed = False
                    # self.button_states = self.command_profile.get_buttons()
                    while not self.se.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                        # self.button_states = self.command_profile.get_buttons()
                    self.se.right_lower_right_switch_pressed = False

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)
            self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            self.logger.save(self.log_filename)
