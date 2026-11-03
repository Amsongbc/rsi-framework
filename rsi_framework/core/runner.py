"""EpisodeRunner: one episode with the assembled blocks.

The lifecycle is the one of window_session.Session, generalised:

  prepare records -> boot environment -> attach VLA client -> attach perception
  -> VLA reset (lease) -> build robot (protocol + tools) -> policy namespace
  -> run configuration -> first observation
  loop: stop / budget / wall-time / evaluator checks -> next turn -> run turn
  close: conclude + verdict -> VLA release -> close recording -> primitive ledger
         -> stop processes -> summary

Blocks that a profile does not use (no VLA, Memory off) are skipped entirely: no
extra calls, observations, model invocations or random draws.
"""
import time
import traceback


class EpisodeRunner:
    def __init__(self, assembly, case, out):
        self.assembly = assembly
        self.case = dict(case)
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        blocks = assembly.blocks
        self.env_block = blocks['environment']
        self.robot = blocks['robot']
        self.vla = blocks.get('vla')
        self.tools = blocks['tools']
        self.protocol = blocks['protocol']
        self.policy = blocks['policy']
        self.memory = blocks['memory']
        self.recorder = blocks['recorder']
        self.policy.prepare_dirs(self)
        self.started = time.monotonic()
        self.ready_at = None
        self.cells = 0
        self.observations = 0
        self.errors = []
        self.env_block.prepare(self)
        self.stop_reason = None
        self.latest = None
        self.context = self.protocol.initial_context(self)

    def boot(self):
        self.recorder.prepare(self)
        self.env_block.boot(self)
        client = self.vla.attach(self) if self.vla is not None else None
        self.context.update(self.env_block.context_entries(self))
        if self.vla is not None:
            self.context['ewam'] = client
        self.tools.attach(self)
        if self.vla is not None:
            self.vla.reset(self)
        self.memory.attach(self)
        self.robo = self.protocol.build_robot(self)
        self.policy.attach(self)
        self.assembly.require_evidence()
        self.recorder.run_config(self)
        self.observe()
        self.ready_at = time.monotonic()

    def observe(self):
        self.latest = self.recorder.snapshot(self)
        return self.latest

    def tool_specs(self):
        return self.tools.specs(self.robo) + self.memory.tool_specs()

    def call_tool(self, name, args):
        if name in {spec.name for spec in self.memory.tool_specs()}:
            return self.memory.call(self, name, args)
        return self.tools.call(self, name, args)

    def loop(self):
        budget_s = self.protocol.wall_budget_s
        while not self.env.episode_over():
            if self.stop_reason:
                break
            if self.env.issued_actions >= self.env.initial_limit:
                self.stop_reason = 'native_budget'
                break
            if time.monotonic() - self.ready_at > budget_s:
                self.stop_reason = 'wall_timeout'
                break
            if not self.env_block.processes_alive(self):
                raise RuntimeError('Evaluation client exited before verdict')
            turn = self.policy.next_turn(self)
            if turn is None:
                if not self.stop_reason:
                    time.sleep(self.policy.idle_s)
                continue
            self.policy.run_turn(self, turn)

    def close(self):
        policy_end = time.monotonic()
        verdict = None
        if self.env_block.booted(self):
            try:
                verdict = self.env_block.finish(self)
            except Exception:
                self.errors.append(traceback.format_exc())
        if self.vla is not None and self.context.get('lease_acquired'):
            try:
                self.vla.release(self)
            except Exception:
                self.errors.append(traceback.format_exc())
        for cleanup in [lambda: self.env.close_recording() if hasattr(self, 'env') else None,
                        lambda: self.robo._ledger.write_jsonl(self.out / 'primitive_calls.jsonl') if hasattr(self, 'robo') else None]:
            try:
                cleanup()
            except Exception:
                self.errors.append(traceback.format_exc())
        self.env_block.stop(self)
        return self.recorder.summarize(self, verdict, policy_end)

    def run(self):
        try:
            self.boot()
            self.loop()
        except BaseException:
            self.errors.append(traceback.format_exc())
            self.stop_reason = 'infrastructure_error'
            print(self.errors[-1], flush=True)
        return self.close()
