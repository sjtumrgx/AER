"""Repository migration checks for the Go2-only neutral package layout."""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROBOT = "go" + "1"
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".lcm",
    ".cpp",
    ".h",
    "",
}


def iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in {".git", ".omx", "__pycache__"} for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            try:
                yield path.relative_to(ROOT).as_posix(), path.read_text(errors="ignore")
            except UnicodeDecodeError:
                continue


class NeutralGo2RepoTest(unittest.TestCase):
    def test_neutral_package_directories_replace_legacy_names(self):
        for dirname in ["gym", "gym_learn", "gym_deploy"]:
            self.assertTrue((ROOT / dirname).is_dir(), dirname)
        legacy_dirs = [
            LEGACY_ROBOT + "_gym",
            LEGACY_ROBOT + "_gym_learn",
            LEGACY_ROBOT + "_gym_deploy",
            f"gym/envs/{LEGACY_ROBOT}",
            f"resources/robots/{LEGACY_ROBOT}",
        ]
        for dirname in legacy_dirs:
            self.assertFalse((ROOT / dirname).exists(), dirname)

    def test_source_no_longer_imports_legacy_package_names(self):
        forbidden = [
            LEGACY_ROBOT + "_gym",
            LEGACY_ROBOT + "_gym_learn",
            LEGACY_ROBOT + "_gym_deploy",
        ]
        offenders = []
        for rel, text in iter_text_files():
            for token in forbidden:
                if token in text:
                    offenders.append(f"{rel}: {token}")
        self.assertEqual([], offenders[:50])

    def test_runtime_entrypoints_are_go2_only(self):
        train = (ROOT / "scripts/train.py").read_text()
        self.assertIn("from gym.envs.go2.go2_config", train)
        self.assertNotIn("--robot", train)
        for script_name in ["play.py", "play_vary_lin.py", "play_vary_ang.py"]:
            script = (ROOT / "scripts" / script_name).read_text()
            self.assertNotIn("--robot", script)
            self.assertIn("AdaptiveGo2ConfigTerrain", script)


    def test_help_entrypoints_expose_smoke_controls_without_isaacgym(self):
        expected = {
            "train.py": ["--iterations", "--num_envs"],
            "play.py": ["--num_steps", "--model_dir"],
            "play_vary_lin.py": ["--num_steps", "--model_dir"],
            "play_vary_ang.py": ["--num_steps", "--model_dir"],
        }
        for script, args in expected.items():
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            for arg in args:
                self.assertIn(arg, result.stdout)


    def test_play_scripts_use_valid_go2_command_defaults(self):
        helper = (ROOT / "scripts" / "play_command_defaults.py").read_text()
        self.assertIn("gait_frequency: float = 3.0", helper)
        self.assertIn("gait_duration: float = 0.5", helper)
        self.assertIn("footswing_height: float = 0.08", helper)
        self.assertIn("stance_width: float = 0.25", helper)

        for script_name in ["play.py", "play_vary_lin.py", "play_vary_ang.py"]:
            script = (ROOT / "scripts" / script_name).read_text()
            self.assertIn("assign_nominal_go2_commands(env.commands", script)
            self.assertNotIn("step_frequency_cmd = 0.0", script)
            self.assertNotIn("footswing_height_cmd = 0.0", script)
            self.assertNotIn("stance_width_cmd = 0.0", script)

    def test_readme_is_go2_focused(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("# Go2 sim-to-real locomotion", readme)
        self.assertIn("gym_deploy/", readme)
        legacy = "Go" + "1"
        forbidden_sections = [
            f"{legacy} examples",
            f"{legacy} deployment flow",
            f"{legacy}/Go2",
            "--robot go2",
        ]
        for section in forbidden_sections:
            self.assertNotIn(section, readme)

    def test_go2_play_rendering_uses_isaacgym_visual_flip(self):
        config = (ROOT / "gym" / "envs" / "go2" / "go2_config.py").read_text()
        self.assertIn("flip_visual_attachments = True", config)

        for script_name in ["play.py", "play_vary_lin.py", "play_vary_ang.py"]:
            script = (ROOT / "scripts" / script_name).read_text()
            self.assertIn("cfg.asset.flip_visual_attachments = True", script)
            self.assertIn("Old checkpoints can persist a stale False value", script)

    def test_go2_asset_references_resolve(self):
        urdf = ROOT / "resources/robots/go2/urdf/go2.urdf"
        xml = ROOT / "resources/robots/go2/xml/go2.xml"
        scene = ROOT / "resources/robots/go2/xml/scene_go2.xml"
        self.assertTrue(urdf.exists())
        self.assertTrue(xml.exists())
        self.assertTrue(scene.exists())

        urdf_text = urdf.read_text()
        self.assertNotIn("package://go2_description", urdf_text)
        for mesh in re.findall(r'filename="([^\"]+\.dae)"', urdf_text):
            self.assertTrue((urdf.parent / mesh).resolve().exists(), mesh)

        xml_text = xml.read_text()
        meshdir_match = re.search(r'<compiler[^>]*meshdir="([^\"]+)"', xml_text)
        mesh_root = xml.parent / (meshdir_match.group(1) if meshdir_match else "")
        for mesh in re.findall(r'file="([^\"]+\.obj)"', xml_text):
            self.assertTrue((mesh_root / mesh).resolve().exists(), mesh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
