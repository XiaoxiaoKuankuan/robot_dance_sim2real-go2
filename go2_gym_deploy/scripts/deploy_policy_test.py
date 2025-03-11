import glob
import pickle as pkl
import lcm
import sys
import pathlib
import torch


from go2_gym_deploy.utils.deployment_runner_test import DeploymentRunner   # 控制四足机器人部署运行  运行器
from go2_gym_deploy.envs.lcm_agent_test import LCMAgent   # 定义智能体函数
from go2_gym_deploy.utils.cheetah_state_estimator_test import StateEstimator  # 订阅底层数据
# from go2_gym_deploy.utils.command_profile import *

# lcm多播通信的标准格式
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def load_and_run_policy(label, experiment_name):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    se = StateEstimator(lc)

    hardware_agent = LCMAgent(se)
    se.spin()
    print('Agent successfully created!')

    policy = load_policy(logdir)
    print('Policy successfully loaded!')

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=se,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)  # 开始实际的实验运行


def load_policy(logdir):

    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')

    def policy(obs, info):
        action = body.forward(obs["obs_history"].to('cpu'))
        return action

    return policy


if __name__ == '__main__':
    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/pretrain-go2/train"

    experiment_name = "example_experiment"

    # default:
    # max_vel=3.5, max_yaw_vel=5.0
    load_and_run_policy(label, experiment_name=experiment_name)
