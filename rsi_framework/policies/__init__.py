"""Policy / Agent blocks: produce the next decision and consume its feedback.

The same command loop serves the external window controller (file mailbox), an
LLM that writes Python cells, and an LLM that returns structured tool / VLA
decisions; what differs is the channel and the operations a profile allows.
"""
