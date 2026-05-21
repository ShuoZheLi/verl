from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import value_decoding.critic_test_metrics_eval as eval_mod


class _DummyChild:
    def __init__(self, name: str, *, alive_after_terminate: bool = False) -> None:
        self.name = name
        self.pid = 12345
        self.terminated = False
        self.killed = False
        self._alive_after_terminate = alive_after_terminate

    def is_alive(self) -> bool:
        if self.killed:
            return False
        if self.terminated:
            return self._alive_after_terminate
        return True

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def join(self, timeout: int) -> None:
        self.join_timeout = timeout


def test_shutdown_vllm_instance_calls_nested_shutdown_methods() -> None:
    llm = SimpleNamespace(
        shutdown=Mock(),
        llm_engine=SimpleNamespace(
            shutdown=Mock(),
            engine_core=SimpleNamespace(close=Mock()),
            model_executor=SimpleNamespace(shutdown=Mock()),
        ),
    )

    eval_mod._shutdown_vllm_instance(llm)

    llm.shutdown.assert_called_once_with()
    llm.llm_engine.shutdown.assert_called_once_with()
    llm.llm_engine.engine_core.close.assert_called_once_with()
    llm.llm_engine.model_executor.shutdown.assert_called_once_with()


def test_terminate_vllm_engine_processes_only_targets_vllm_children(monkeypatch) -> None:
    engine_child = _DummyChild("EngineCore_DP0")
    stubborn_child = _DummyChild("vllm-worker", alive_after_terminate=True)
    unrelated_child = _DummyChild("pytest-worker")
    monkeypatch.setattr(
        eval_mod.multiprocessing,
        "active_children",
        lambda: [engine_child, stubborn_child, unrelated_child],
    )

    eval_mod._terminate_vllm_engine_processes()

    assert engine_child.terminated
    assert not engine_child.killed
    assert stubborn_child.terminated
    assert stubborn_child.killed
    assert not unrelated_child.terminated
    assert not unrelated_child.killed
