"""ContextBuilder: what an LLM channel sees each turn.

Only public information enters: the task instruction, the latest observation (robot
state, budget, camera image files), the tools and operations this profile allows,
the protocol's own rules, recent commands with their outputs or errors, and the
Memory text when the Memory block offers it. No hidden environment state.
"""
import json

DEFAULT_SYSTEM = '''You control a robot episode one command per turn.
Reply with exactly one JSON object and nothing else:
{"op": <one of OPS>, "note": "<what you see and why>", ...}
  python: {"code": "<Python using robo.*>"}       tool: {"tool": "<name>", "args": {...}}
  vla: {"language": "<short instruction or null>", "model_actions": <1..N>}
  observe: {}     finish: {}
Add "expected_change": "<visible effect you expect>" to motion commands.
Base every decision on the current images and state; a returned function value is not
proof that an object moved. Finish when the task is complete or cannot progress.'''


def _truncate(value, limit=1500):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + ' ...[truncated]'


def build(run, policy, identifier):
    latest = run.latest or {}
    specs = [spec.public() for spec in run.tool_specs()]
    system = (policy.system_prompt or DEFAULT_SYSTEM).replace('<one of OPS>', ' | '.join(policy.ops))
    rules = getattr(run.protocol, 'describe', lambda: {})()
    memory_text = run.memory.context_text(run)
    recent = [{'turn': row['turn'], 'command': row['command'], 'output': _truncate(row['output']),
               'error': _truncate(row['error'].strip().splitlines()[-1]) if row.get('error') else None}
              for row in policy.history[-policy.history_turns:]]
    user = {'task': latest.get('instruction'), 'turn': identifier,
            'observation': {k: latest.get(k) for k in ('observation_index', 'native_action_index', 'step_limit',
                                                        'native_budget_remaining', 'robot_state')},
            'protocol': rules, 'tools': specs, 'recent': recent}
    if memory_text:
        user['memory'] = memory_text
    images = []
    if policy.images == 'latest' and latest.get('images'):
        images = [{'camera': camera, 'path': path, 'sha256': latest['image_sha256'][camera]}
                  for camera, path in latest['images'].items()]
    return {'id': identifier, 'messages': [{'role': 'system', 'content': system},
                                           {'role': 'user', 'content': json.dumps(user, ensure_ascii=False, default=str)}],
            'images': images}
