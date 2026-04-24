# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "resolve_model_init_path.py"
SPEC = importlib.util.spec_from_file_location("resolve_model_init_path", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_hf_model(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("", encoding="utf-8")


class ResolveModelInitPathTests(unittest.TestCase):
    def test_normalize_path_strips_trailing_slashes(self):
        self.assertEqual(MODULE.normalize_path("/tmp/foo/"), "/tmp/foo")
        self.assertEqual(MODULE.normalize_path("/"), "/")
        self.assertEqual(MODULE.normalize_path("hdfs://bucket/path//"), "hdfs://bucket/path")

    def test_plan_passthrough_for_hf_model_id(self):
        plan = MODULE.plan_model_init_path("Qwen/Qwen2.5-7B", "actor")
        self.assertEqual(plan.mode, "resolved")
        self.assertEqual(plan.resolved_path, "Qwen/Qwen2.5-7B")

    def test_absolute_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            MODULE.plan_model_init_path("/definitely/missing/model/path", "actor")

    def test_missing_relative_local_path_raises_when_parent_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "models").mkdir()
            with self.assertRaises(FileNotFoundError):
                MODULE.plan_model_init_path(str(root / "models" / "missing"), "actor")

    def test_resolve_direct_hf_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "hf_model"
            write_hf_model(model_dir)

            plan = MODULE.plan_model_init_path(str(model_dir), "actor")
            self.assertEqual(plan.mode, "resolved")
            self.assertEqual(plan.resolved_path, str(model_dir))

    def test_resolve_merged_hf_container_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_root = Path(tmpdir) / "merged_hf"
            actor_dir = merged_root / "actor"
            write_hf_model(actor_dir)

            plan = MODULE.plan_model_init_path(str(merged_root), "actor")
            self.assertEqual(plan.mode, "resolved")
            self.assertEqual(plan.resolved_path, str(actor_dir))

    def test_resolve_raw_component_huggingface_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor_dir = Path(tmpdir) / "actor"
            actor_dir.mkdir(parents=True, exist_ok=True)
            (actor_dir / "fsdp_config.json").write_text("{}", encoding="utf-8")
            write_hf_model(actor_dir / "huggingface")

            plan = MODULE.plan_model_init_path(str(actor_dir), "actor")
            self.assertEqual(plan.mode, "resolved")
            self.assertEqual(plan.resolved_path, str(actor_dir / "huggingface"))

    def test_resolve_global_step_root_with_existing_merged_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "global_step_950"
            actor_dir = root / "actor"
            (actor_dir / "fsdp_config.json").parent.mkdir(parents=True, exist_ok=True)
            (actor_dir / "fsdp_config.json").write_text("{}", encoding="utf-8")
            (actor_dir / "huggingface").mkdir()
            (actor_dir / "huggingface" / "config.json").write_text("{}", encoding="utf-8")
            write_hf_model(root / "merged_hf" / "actor")

            plan = MODULE.plan_model_init_path(str(root), "actor")
            self.assertEqual(plan.mode, "resolved")
            self.assertEqual(plan.resolved_path, str(root / "merged_hf" / "actor"))

    def test_merge_raw_component_when_only_metadata_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actor_dir = Path(tmpdir) / "actor"
            (actor_dir / "fsdp_config.json").parent.mkdir(parents=True, exist_ok=True)
            (actor_dir / "fsdp_config.json").write_text("{}", encoding="utf-8")
            (actor_dir / "huggingface").mkdir()
            (actor_dir / "huggingface" / "config.json").write_text("{}", encoding="utf-8")
            log_dir = Path(tmpdir) / "logs"

            def fake_run(command, check, stdout, stderr):
                del command, check, stdout, stderr
                write_hf_model(actor_dir.parent / "merged_hf" / "actor")
                return None

            with patch.object(MODULE.subprocess, "run", side_effect=fake_run) as mock_run:
                resolved = MODULE.ensure_model_init_path(str(actor_dir), "actor", log_dir=str(log_dir))

            self.assertEqual(resolved, str(actor_dir.parent / "merged_hf" / "actor"))
            self.assertTrue((log_dir / "model_merge_actor.log").exists())
            self.assertEqual(mock_run.call_count, 1)

    def test_raw_component_missing_huggingface_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            critic_dir = Path(tmpdir) / "critic"
            critic_dir.mkdir(parents=True, exist_ok=True)
            (critic_dir / "fsdp_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                MODULE.plan_model_init_path(str(critic_dir), "critic")


if __name__ == "__main__":
    unittest.main()
