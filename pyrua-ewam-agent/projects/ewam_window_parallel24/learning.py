"""Pure frozen-memory validation only. Current-window reasoning supplies learning."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import math
import json

def gate(scores, threshold=30):
    if len(scores) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError('Gate requires three valid native scores in [0,1]')
    return 'direct' if math.fsum(scores)/3*100 < threshold else 'hybrid'


def validate_memory(memory, task):
    if not isinstance(memory, dict) or memory.get('task') != task:
        raise ValueError('Experience task mismatch')
    lessons = memory.get('lessons')
    if not isinstance(lessons, list) or len(lessons) > 10:
        raise ValueError('Experience must have at most 10 lessons')
    for lesson in lessons:
        for key in ('trigger', 'diagnosis', 'adaptation', 'verification', 'confidence', 'source_episode'):
            if not isinstance(lesson.get(key), str) or not lesson[key].strip():
                raise ValueError('Malformed experience field ' + key)
        if lesson['confidence'] not in ('observed', 'hypothesis'):
            raise ValueError('Experience confidence must distinguish observations and hypotheses')
    if len(json.dumps(memory)) > 16000:
        raise ValueError('Experience is too large')
