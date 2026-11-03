"""VLA plugins: obs + language -> a chunk of native actions.

A VLA block provides
  contract   {'cameras', 'state_keys', 'action_keys': {key: size}, 'stride', ...}; checked
             against the RobotSpec at assembly, so a mismatch is reported before any run
  attach(run) -> client or None, reset(run), release(run)
  infer(run, obs, language) -> {'actions': [native action dicts], 'instruction': str}
  describe()
For a model you can call from Python, subclass vla_backends.base.InProcessVLA and
implement predict(images, state, language). A model behind a server needs its own
client block (ewam.py is one). Each model's own conversion (EWAM: 16 model -> 48
native actions) stays inside its block.
"""
