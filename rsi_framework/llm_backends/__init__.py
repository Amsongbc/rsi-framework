"""LLMBackend blocks: where commands come from.

- command_queue: an external controller (the window AI, a person or another program)
  writes commands / replies as files; no API is called and no usage is invented.
- http_chat: an OpenAI-compatible chat endpoint; base URL, model and key file come
  only from the profile.
- scripted: fixed replies for replay and tests (evidence label `replay`).
"""
