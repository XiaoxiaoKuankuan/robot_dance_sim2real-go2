import glob
import pathlib
import sys
import time
import torch
import lcm
from go2_gym_deploy.envs.lcm_agent_test import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator_test import StateEstimator

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

# 配置模型路径字典
MODEL_PATHS = {
    0: './model/go2/stand_2025-03-18_19-19-44.jit',
    1: './model/go2/stand_2025-03-17_08-46-33.jit',
    2: './model/go2/swing_2025-03-18_15-40-36.jit',
    3: './model/go2/swing_2025-03-17_08-49-23.jit'
}


def load_and_run_policy(experiment_name="default_experiment"):
    # 初始化状态估计器
    se = StateEstimator(lc)

    try:
        # 预加载所有策略模型
        policy_dict = {k: torch.jit.load(v) for k, v in MODEL_PATHS.items()}
        for m in policy_dict.values():
            m.eval()
        print("所有策略模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 初始化硬件代理
    hardware_agent = LCMAgent(se)
    se.spin()

    # 等待用户按下 R2 才开始
    print("等待按下 R2 键以启动策略...")
    while True:
        button_states = se.get_buttons()
        if se.right_lower_right_switch_pressed:
            print(">>>>>>>>>>>>>>> R2 被按下，开始执行默认策略 <<<<<<<<<<<<<")
            se.right_lower_right_switch_pressed = False  # 清除按键状态
            break
        time.sleep(0.1)

    # 主控制循环
    obs = hardware_agent.get_obs()
    try:
        current_policy = policy_dict[0]  # 默认策略
        current_model_id = 0
        last_switch_time = time.time()

        while True:
            start_time = time.time()

            # 安全获取当前模型ID
            new_model_id = se.get_current_model_id()
            if new_model_id not in policy_dict:
                new_model_id = current_model_id

            # 切换策略（冷却时间1秒）
            if (new_model_id != current_model_id):
                current_model_id = new_model_id
                current_policy = policy_dict[current_model_id]
                last_switch_time = time.time()
                print(f"\n--- 切换到策略 {current_model_id} ---")

            # 生成控制指令
            action = current_policy(obs)

            # 执行动作
            obs, ret, done, info = hardware_agent.step(action)

            # 控制循环频率
            elapsed = time.time() - start_time
            if elapsed < 0.02:  # 50Hz
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断...")
    finally:
        print("系统已关闭")


if __name__ == "__main__":
    load_and_run_policy(experiment_name="multi_policy_test")