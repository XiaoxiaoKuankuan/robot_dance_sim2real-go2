import math
import select
import threading
import time

import numpy as np

from go2_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go2_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go2_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt

# 将四元数（quaternion）转换为欧拉角（Roll, Pitch, Yaw，简称 RPY）
def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


# 根据欧拉角计算旋转矩阵
def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class StateEstimator:
    def __init__(self, lc, use_cameras=False):  # defaul use_cameras=True

        # 这里腿的顺序为什么要转换？ 对应实物 实物是 RF LF RH LH 训练网络出来是LF RF LH  RH
        # reverse legs
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.lc = lc
        self._current_model_id = 0  # 默认模型ID
        self.joint_pos = np.zeros(12)  # 关节位置
        self.joint_vel = np.zeros(12)  # 关节速度
        self.tau_est = np.zeros(12)  # 估计的关节力矩
        self.world_lin_vel = np.zeros(3)  # 世界坐标系下的线速度
        self.world_ang_vel = np.zeros(3)  # 世界坐标系下的角速度
        self.euler = np.zeros(3)  # 欧拉角 (Roll, Pitch, Yaw)
        self.R = np.eye(3)  # 旋转矩阵 (3x3单位矩阵)
        self.buf_idx = 0  # 缓存索引

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.base_ang_vel_w = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        self.init_time = time.time()
        self.received_first_legdata = False
        # 使用 LCM订阅不同的消息通道，并指定回调函数来处理接收到的消息
        # 这段代码使用 LCM 订阅了 IMU、腿部控制、遥控指令 三个频道。
        # 每个订阅都会在收到数据时 自动调用回调函数 进行处理。
        # 这是一种 事件驱动 的通信模式
        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)

        self.body_loc = np.array([0, 0, 0])
        self.body_quat = np.array([0, 0, 0, 1])

    def get_body_linear_vel(self):
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_body_angular_vel(self):
        self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / (self.dt_history+0.000001), axis=0) + (
                1 - self.smoothing_ratio) * self.body_ang_vel
        print("微分得到的self.body_ang_vel is :", self.body_ang_vel)
        # self.body_ang_vel = self.base_ang_vel_w
        
        # print("self.dt_history:", self.dt_history)
        # print("self.deuler_history:", self.deuler_history)

        return self.body_ang_vel

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav


    def get_rpy(self):
        return self.euler

    def get_current_model_id(self):
        """获取当前选择的模型ID"""
        return self._current_model_id

    def get_buttons(self):
        return np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch,
                         self.right_upper_switch])

    def get_dof_pos(self):
        print("dofposquery", self.joint_pos[self.joint_idxs])
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]

    def get_body_loc(self):
        return np.array(self.body_loc)

    def get_body_quat(self):
        return np.array(self.body_quat)

    # 三个 LCM 订阅回调函数
    def _legdata_cb(self, channel, data):
        # print("update legdata")
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata: {time.time() - self.init_time}")

        msg = leg_control_data_lcmt.decode(data)
        # print(msg.q)
        self.joint_pos = np.array(msg.q)  # 关节位置
        self.joint_vel = np.array(msg.qd)  # 关节速度
        self.tau_est = np.array(msg.tau_est)  # 估算的关节力矩
        # print(f"update legdata {msg.id}")

    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)
        # print("msg.rpy:", msg.rpy)
        self.euler = np.array(msg.rpy)

        self.R = get_rotation_matrix_from_rpy(self.euler)  # # 计算旋转矩阵

        self.contact_state = 1.0 * (np.array(msg.contact_estimate) > 200)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev  # 存储 欧拉角变化量
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev # 存储 时间间隔

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)

        # self.base_ang_vel_w = msg.omegaWorld  # 直接获取机身角速度
        # print("直接获取的base_ang_vel_w is :", self.base_ang_vel_w)


    def _rc_command_cb(self, channel, data):

        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = (
                    (msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((
                                                           msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((
                                                            msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = (
                    (msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((
                                                            msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((
                                                             msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        # 模型选择逻辑
        if 0 <= msg.mode <= 3:  # 有效范围检查
            new_id = int(msg.mode)
            if new_id != self._current_model_id:
                self._current_model_id = new_id
                print(f"\n[Model Switch] Current Model ID: {new_id}")
        # print(self.right_stick, self.left_stick)

    # --------------------------------------------------

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    se.poll()
