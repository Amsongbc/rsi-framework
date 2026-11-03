"""ARX X5 bimanual robot as RoboDojo / PyRUA publish it (the original cohorts' robot)."""
from rsi_framework.core.registry import register
from rsi_framework.robots.spec import RobotSpec

ARX_X5 = RobotSpec(
    name='arx_x5',
    cameras=('cam_head', 'cam_left_wrist', 'cam_right_wrist'),
    state_keys=('left_arm_joint_state', 'left_ee_joint_state', 'left_ee_pose',
                'right_arm_joint_state', 'right_ee_joint_state', 'right_ee_pose'),
    action_keys={'left_arm_joint_state': 6, 'left_ee_joint_state': 1,
                 'right_arm_joint_state': 6, 'right_ee_joint_state': 1},
    description='Two 6-joint arms with grippers; absolute joint targets; three RGB cameras')


@register('robot', 'arx_x5', aliases=('arx_x5_bimanual',), description='ARX X5 bimanual (RoboDojo arx_x5 suite)')
def make(**options):
    return ARX_X5
