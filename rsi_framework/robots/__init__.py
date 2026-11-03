"""Robot plugins: what the body is.

A RobotSpec names the cameras an environment publishes, the state keys, and the
native action keys with their sizes. Recorders, tools, VLA adapters, datasets and
trainers read these names from the spec instead of hard-coding a robot, and the
assembly checks that environment, VLA and tools agree on them.
"""
