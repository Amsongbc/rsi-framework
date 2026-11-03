"""Public observations, native actions and synchronized three-camera video."""
from __future__ import annotations

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os

import json
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

RUNTIME = Path(_export_os.environ.get('PROJECTS_ROOT', str(Path(__file__).resolve().parents[1])))/'ewam_comparison'
sys.path.insert(0,str(RUNTIME))
from session import CountingEnv, plain, write_json
from ewam_service import CAMERAS


def append_json(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(plain(row),ensure_ascii=False)+'\n')


class RecordedEnv(CountingEnv):
    def __init__(self, env, out):
        super().__init__(env)
        self.out = Path(out)
        self.videos = {}
        self.frame_index = 0
        self.first_native_owner = None
        self.initial_limit = int(self.get_status()['step_limit'])
        (self.out/'native_video').mkdir()
        self.record(self.get_obs(), action=None)

    def record(self, obs, action):
        status = self.get_status()
        for camera in CAMERAS:
            rgb = np.asarray(obs['vision'][camera]['color'])[...,:3].astype(np.uint8)
            if camera not in self.videos:
                h,w = rgb.shape[:2]
                writer = cv2.VideoWriter(str(self.out/'native_video'/f'{camera}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
                if not writer.isOpened():
                    raise RuntimeError('Could not open native video writer')
                self.videos[camera] = writer
            self.videos[camera].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        row = {'time':time.time(),'frame_index':self.frame_index,
               'native_step':int(status.get('step',0)), 'issued_actions':self.issued_actions,
               'owner':'initial' if action is None else self.owner,
               'action':action, 'robot_state':obs.get('state',{}),
               'instruction':str(obs.get('instruction') or obs.get('instructions') or '')}
        append_json(self.out/'native_actions.jsonl',row)
        write_json(self.out/'progress.json',{'phase':'running','native_step':row['native_step'],
                                            'initial_native_limit':self.initial_limit,
                                            'issued_actions':self.issued_actions,
                                            'owner':row['owner'],'updated_at':row['time']})
        self.frame_index += 1

    def step(self, action):
        result = super().step(action)
        if self.first_native_owner is None:self.first_native_owner=self.owner
        self.record(self.get_obs(),action)
        return result

    def close_recording(self):
        for writer in self.videos.values():
            writer.release()
        write_json(self.out/'native_recording_summary.json',{
            'frames_per_camera':self.frame_index,'video_fps':30,
            'frame_time_meaning':'One frame per native action plus initial frame; playback fps is not simulation wall-clock timing',
            'native_actions_issued':self.issued_actions,'by_owner':self.actions_by_owner,
            'initial_native_limit':self.initial_limit})


class RecordedSam3:
    """Preserve actual perception input/mask without changing the returned result."""
    def __init__(self, client, context):
        self.client=client
        self.context=context
        self.index=0
        self.out=context['out']/'perception'
        (self.out/'images').mkdir(parents=True,exist_ok=True)
        (self.out/'masks').mkdir(exist_ok=True)

    def __getattr__(self,name):
        return getattr(self.client,name)

    def segment(self,image,*,text_prompt=None,point=None,min_score=.2):
        self.index+=1
        rgb=np.ascontiguousarray(np.asarray(image)[...,:3].astype(np.uint8))
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        path=self.out/'images'/(digest+'.png')
        if not path.exists():
            from PIL import Image
            Image.fromarray(rgb).save(path)
        env=self.context.get('env')
        row={'index':self.index,'time':time.time(),'text_prompt':text_prompt,
             'point':point,'min_score':min_score,'input_image':str(path.relative_to(self.context['out'])),
             'native_step':None if env is None else env.get_status().get('step')}
        tick=time.monotonic()
        try:
            result=self.client.segment(image,text_prompt=text_prompt,point=point,min_score=min_score)
            row.update(found=result.found,score=result.score,box=result.box,reason=result.reason)
            if result.mask is not None:
                mask=self.out/'masks'/f'{self.index:05d}.npz'
                np.savez_compressed(mask,mask=result.mask)
                row['mask']=str(mask.relative_to(self.context['out']))
            return result
        except BaseException as exc:
            row['error']=type(exc).__name__+': '+str(exc)
            raise
        finally:
            row['wall_s']=time.monotonic()-tick
            append_json(self.out/'segmentation.jsonl',row)
