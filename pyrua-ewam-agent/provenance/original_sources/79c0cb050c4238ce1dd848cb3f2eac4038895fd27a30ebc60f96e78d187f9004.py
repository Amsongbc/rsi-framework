"""Protocol tests with explicit stubs; these are never simulator experiment results."""
import unittest

import numpy as np

from ewam_service import CAMERAS, JOINT_KEYS, public_observation
from session import CountingEnv, execute_ewam


def observation(step=0):
    return {"vision": {camera: {"color": np.zeros((2, 2, 3), dtype=np.uint8),
                                "depth": np.ones((2, 2)), "segmentation": "private"}
                       for camera in CAMERAS},
            "state": {**{key: np.array([step], dtype=np.float32) for key in JOINT_KEYS},
                      "object_pose": "private"},
            "instruction": "Original instruction", "reward": 1,
            "task_answer": "private"}


class Env:
    def __init__(self, limit=200):
        self.step_count, self.limit = 0, limit

    def get_obs(self):
        return observation(self.step_count)

    def get_status(self):
        return {"step": self.step_count, "step_limit": self.limit}

    def step(self, action):
        self.step_count += 1
        return self.get_obs(), 0, self.episode_over(), {}

    def episode_over(self):
        return self.step_count >= self.limit


class Client:
    def __init__(self):
        self.observations = []

    def call(self, op, **kwargs):
        self.observations.append(kwargs["obs"])
        return {"actions": [{"action": index} for index in range(48)],
                "inference_s": 0.0, "instruction": kwargs["obs"]["instruction"]}


class EpisodeContract(unittest.TestCase):
    def test_rgb_and_joint_only_no_evaluator_or_object_fields(self):
        original = observation()
        public = public_observation(original, "Subtask")
        self.assertEqual(set(public), {"vision", "state", "instruction"})
        self.assertEqual(set(public["state"]), set(JOINT_KEYS))
        self.assertEqual(public["instruction"], "Subtask")
        self.assertEqual(original["instruction"], "Original instruction")
        self.assertTrue(all(set(view) == {"color"} for view in public["vision"].values()))
        public["vision"]["cam_head"]["color"][0, 0, 0] = 255
        self.assertEqual(original["vision"]["cam_head"]["color"][0, 0, 0], 0)

    def test_full_chunks_reobserve_after_48_native_steps(self):
        env, client = CountingEnv(Env(limit=96)), Client()
        reports = execute_ewam(env, client, "episode", chunks=2)
        self.assertEqual([r["executed_native_actions"] for r in reports], [48, 48])
        self.assertEqual([int(o["state"][JOINT_KEYS[0]][0]) for o in client.observations], [0, 48])
        self.assertEqual(env.issued_actions, 96)

    def test_instruction_container_is_not_stringified_as_a_python_list(self):
        obs = observation()
        obs["instruction"] = [b"Original instruction"]
        self.assertEqual(public_observation(obs)["instruction"], "Original instruction")

    def test_primitive_and_policy_actions_share_one_budget(self):
        env, client = CountingEnv(Env(limit=6)), Client()
        for _ in range(4):
            env.step({"primitive": True})
        env.owner = "ewam"
        execute_ewam(env, client, "episode")
        self.assertEqual(env.issued_actions, 6)
        self.assertEqual(env.actions_by_owner, {"primitive": 4, "ewam": 2})

    def test_native_limit_stops_midchunk_without_overspending(self):
        env, client = CountingEnv(Env(limit=5)), Client()
        reports = execute_ewam(env, client, "episode", chunks=4)
        self.assertEqual(env.issued_actions, 5)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["executed_native_actions"], 5)

    def test_shorter_chunk_preserves_stride_three(self):
        env, client = CountingEnv(Env()), Client()
        execute_ewam(env, client, "episode", chunks=2, model_actions=4)
        self.assertEqual(env.issued_actions, 24)
        self.assertEqual([int(o["state"][JOINT_KEYS[0]][0]) for o in client.observations], [0, 12])

    def test_finished_episode_does_not_call_policy(self):
        env, client = CountingEnv(Env(limit=0)), Client()
        self.assertEqual(execute_ewam(env, client, "episode"), [])
        self.assertEqual(client.observations, [])

    def test_bad_chunk_sizes_are_rejected_before_inference(self):
        for kw in [{"chunks": 0}, {"chunks": 5}, {"model_actions": 0}, {"model_actions": 17}]:
            client = Client()
            with self.assertRaises(ValueError):
                execute_ewam(CountingEnv(Env()), client, "episode", **kw)
            self.assertEqual(client.observations, [])


if __name__ == "__main__":
    unittest.main()
