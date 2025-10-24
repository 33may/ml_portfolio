# comments in English only
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaaclab import sim as sim_utils

@configclass
class CollectCubesEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    action_space = 7       # number of controllable joints (for Franka)
    observation_space = 14  # e.g. joint positions + velocities
    state_space = 0        # optional global state, keep 0 if unused

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # # mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    #     ),
    # )

    # robot
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.fix_base = True
    # robot_cfg.ee_frame_name = "panda_hand_tcp"

    # scene — ONLY the supported args
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.5,
        replicate_physics=True,
    )
