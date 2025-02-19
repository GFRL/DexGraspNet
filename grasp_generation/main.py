"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: Entry of the program
"""

import os

os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import transforms3d.quaternions as tq


# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--object_code', default=
    "core_bottle_13544f09512952bbc9273c10871e1c3d",
            # "core_pillow_f3833476297f19c664b3b9b23ddfcbc",
            # "sem_Shampoo_2956aa89a775941267510767535a263a",
            # "sem_Camera_ebe5e14f4975e9029174750cc2a009c3",
            # "sem_Snowman_e57005090a392849320d4939a0975e7d",
            # "mujoco_AMBERLIGHT_UP_W",
    type=str)
parser.add_argument('--name', default='exp_33', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=80, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
# hyper parameters
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--noise_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=30.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.2, type=float)
parser.add_argument('--distance_upper', default=0.3, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

# Trail_id
parser.add_argument('--Trail_id', default=0, type=int)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# prepare models
object_code_list = [args.object_code]
total_batch_size = len(object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)
hand_model = HandModel(
    urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
    contact_points_path='allegro_hand_description/contact_points.json', 
    n_surface_points=1000, 
    device=device
)

object_model = ObjectModel(
    data_root_path='../assets/DGNObj',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(object_code_list)

initialize_convex_hull(hand_model, object_model, args)

print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()

optim_config = {
    'switch_possibility': args.switch_possibility,
    'starting_temperature': args.starting_temperature,
    'temperature_decay': args.temperature_decay,
    'annealing_period': args.annealing_period,
    'noise_size': args.noise_size,
    'stepsize_period': args.stepsize_period,
    'mu': args.mu,
    'device': device
}
optimizer = Annealing(hand_model, **optim_config)

# try:
#     shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
# except FileNotFoundError:
#     pass
# os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
# logger_config = {
#     'thres_fc': args.thres_fc,
#     'thres_dis': args.thres_dis,
#     'thres_pen': args.thres_pen
# }
# logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)
energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
# logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]

        # logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]
# try:
#     shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
# except FileNotFoundError:
#     pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        # qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        # euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        quat = tq.mat2quat(rot.numpy())
        # qpos.update(dict(zip(rot_names, euler)))
        # qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        grasp_qpos = np.concatenate([hand_pose[:3].numpy(), quat, hand_pose[9:].numpy()])
        save_dict = {
            "obj_scale": scale,
            "obj_pose": np.array([0, 0, 0, 1.0, 0, 0, 0]),
            'obj_path': os.path.join("assets/DGNObj",object_code_list[i]),
            "grasp_qpos": grasp_qpos,
            "grasp_error": energy[idx].item(),
        }
        save_path_dir = os.path.join(result_path, object_code_list[i],f"scale{int(scale*100):03d}")
        os.makedirs(save_path_dir, exist_ok=True)
        save_path = os.path.join(save_path_dir, f"{len(os.listdir(save_path_dir))}.npy")
        np.save(save_path, save_dict, allow_pickle=True)
        # hand_pose = hand_pose_st[idx].detach().cpu()
        # qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
        # rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        # euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        # qpos_st.update(dict(zip(rot_names, euler)))
        # qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        # data_list.append(dict(
        #     scale=scale,
        #     qpos=qpos,
        #     qpos_st=qpos_st,
        #     energy=energy[idx].item(),
        #     E_fc=E_fc[idx].item(),
        #     E_dis=E_dis[idx].item(),
        #     E_pen=E_pen[idx].item(),
        #     E_spen=E_spen[idx].item(),
        #     E_joints=E_joints[idx].item(),
        # ))
    # np.save(os.path.join(result_path, object_code_list[i] + '.npy'), data_list, allow_pickle=True)
