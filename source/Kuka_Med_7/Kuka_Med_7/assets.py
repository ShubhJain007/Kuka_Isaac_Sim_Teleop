
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Define the Kuka Med 7 configuration
MED7_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/kneepolean/Isaac_Lab_projects/Kuka_Med_7/med7.urdf",
        fix_base=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos={
            "lbr_A1": 0.0,
            "lbr_A2": 0.0,
            "lbr_A3": 0.0,
            "lbr_A4": 0.0,
            "lbr_A5": 0.0,
            "lbr_A6": 0.0,
            "lbr_A7": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["lbr_A.*"],
            stiffness=400.0,
            damping=80.0,
        ),
    },
)
