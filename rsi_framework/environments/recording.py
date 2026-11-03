"""Shared native budget and recording for framework environments.

Writes the same files as the original RecordedEnv (native_actions.jsonl with one row
per native action after it ran, native_video/<camera>.mp4, progress.json,
native_recording_summary.json), for any RobotSpec cameras. The PyRUA adapter keeps
the original RecordedEnv so compat runs are unchanged.
"""
from pathlib import Path
import time

import numpy as np

from rsi_framework.core.contracts import EpisodeEnded
from rsi_framework.core.io import append_json, write_json


class RecordedNativeEnv:
    vla_owner = 'vla'

    def __init__(self, env, out, cameras):
        self.env, self.out, self.cameras = env, Path(out), tuple(cameras)
        self.issued_actions = 0
        self.owner = 'primitive'
        self.actions_by_owner = {'primitive': 0, 'vla': 0}
        self.first_native_owner = None
        self.initial_limit = int(env.get_status()['step_limit'])
        self.videos, self.frame_index = {}, 0
        (self.out / 'native_video').mkdir()
        self.record(env.get_obs(), None)

    def get_obs(self):
        return self.env.get_obs()

    def get_status(self):
        return self.env.get_status()

    def episode_over(self):
        return self.env.episode_over()

    def conclude(self):
        return self.env.conclude()

    def step(self, action):
        if max(int(self.get_status().get('step', 0)), self.issued_actions) >= self.initial_limit:
            raise EpisodeEnded('Shared native action budget exhausted')
        self.issued_actions += 1
        self.actions_by_owner[self.owner] += 1
        result = self.env.step(action)
        if self.first_native_owner is None:
            self.first_native_owner = self.owner
        self.record(self.get_obs(), action)
        return result

    def record(self, obs, action):
        import cv2
        for camera in self.cameras:
            rgb = np.asarray(obs['vision'][camera]['color'])[..., :3].astype(np.uint8)
            if camera not in self.videos:
                h, w = rgb.shape[:2]
                writer = cv2.VideoWriter(str(self.out / 'native_video' / f'{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Could not open native video writer')
                self.videos[camera] = writer
            self.videos[camera].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        status = self.get_status()
        row = {'time': time.time(), 'frame_index': self.frame_index, 'native_step': int(status.get('step', 0)),
               'issued_actions': self.issued_actions, 'owner': 'initial' if action is None else self.owner,
               'action': action, 'robot_state': obs.get('state', {}), 'instruction': str(obs.get('instruction') or '')}
        append_json(self.out / 'native_actions.jsonl', row)
        write_json(self.out / 'progress.json', {'phase': 'running', 'native_step': row['native_step'],
                                                'initial_native_limit': self.initial_limit,
                                                'issued_actions': self.issued_actions, 'owner': row['owner'],
                                                'updated_at': row['time']})
        self.frame_index += 1

    def close_recording(self):
        for writer in self.videos.values():
            writer.release()
        write_json(self.out / 'native_recording_summary.json', {
            'frames_per_camera': self.frame_index, 'video_fps': 30,
            'frame_time_meaning': 'One frame per native action plus initial frame', 'native_actions_issued': self.issued_actions,
            'by_owner': self.actions_by_owner, 'initial_native_limit': self.initial_limit})
