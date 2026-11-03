#!/usr/bin/env python3
"""Audit paired initial observations and common EWAM action prefixes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
rows = []
for task in ['push_T', 'cover_blocks', 'build_tower']:
    b = root/'runs'/f'baseline_{task}_s0_l0'
    h = root/'runs'/f'hybrid_{task}_s0_l0'
    if not all((p/'observations/0000/observation.json').exists() for p in [b,h]):
        continue
    configs = [json.loads((p/'run_config.json').read_text()) for p in [b,h]]
    obs = [json.loads((p/'observations/0000/observation.json').read_text()) for p in [b,h]]
    keys = ['task','eval_seed','layout','EWAM_seed','observation_protocol','step_limit','sim_gpu']
    row = {'task':task,'config_equal':{k:configs[0][k]==configs[1][k] for k in keys},
           'instruction_equal':obs[0]['instruction']==obs[1]['instruction'],'images':{}}
    for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
        paths = [p/'observations/0000'/f'{cam}.png' for p in [b,h]]
        arrays = [np.asarray(Image.open(p)).astype(float) for p in paths]
        row['images'][cam] = {'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                                  'exact_equal':bool(np.array_equal(*arrays)),
                                  'mean_absolute_pixel_difference':float(np.mean(np.abs(arrays[0]-arrays[1])))}
    streams = []
    for p in [b,h]:
        events = [json.loads(line) for line in (p/'events.jsonl').read_text().splitlines()]
        streams.append([e['action'] for e in events if e['kind']=='ewam_action'][:48])
    row['first_48_ewam_actions_equal'] = len(streams[0])==48 and streams[0]==streams[1]
    if len(streams[0])==len(streams[1])==48:
        values = [np.array([sum((action[k] for k in sorted(action)),[]) for action in stream]) for stream in streams]
        row['first_48_action_mean_absolute_difference'] = float(np.mean(np.abs(values[0]-values[1])))
        row['first_48_action_max_absolute_difference'] = float(np.max(np.abs(values[0]-values[1])))
    row['initial_robot_state_equal'] = obs[0]['robot_state']==obs[1]['robot_state']
    rows.append(row)
(root/'paired_initial_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
