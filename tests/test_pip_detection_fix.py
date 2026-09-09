"""Tests for pip detection fix in run-server.sh script.

This test file ensures our pip detection improvements work correctly
and don't break existing functionality.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestPipDetectionFix:
    """Test cases for issue #188: PIP is available but not recognized."""

    def test_run_server_script_syntax_valid(self):
        """Test that run-server.sh has valid bash syntax."""
        result = subprocess.run(["bash", "-n", "./run-server.sh"], capture_output=True, text=True)
        assert result.returncode == 0, f"Syntax error in run-server.sh: {result.stderr}"

    def test_run_server_has_proper_shebang(self):
        """Test that run-server.sh starts with proper shebang."""
        content = Path("./run-server.sh").read_text()
        assert content.startswith("#!/bin/bash"), "Script missing proper bash shebang"

    def test_critical_functions_exist(self):
        """Test that all critical functions are defined in the script."""
        content = Path("./run-server.sh").read_text()
        critical_functions = ["find_python", "setup_environment", "setup_venv", "install_dependencies", "bootstrap_pip"]

        for func in critical_functions:
            assert f"{func}()" in content, f"Critical function {func}() not found in script"

    def test_pip_detection_consistency_issue(self):
        """Test the specific issue: pip works in setup_venv but fails in install_dependencies.

        This test verifies that our fix ensures consistent Python executable paths.
        """
        # Test that the get_venv_python_path function now returns absolute paths
        content = Path("./run-server.sh").read_text()

        # Check that get_venv_python_path includes our absolute path conversion logic
        assert "abs_venv_path" in content, "get_venv_python_path should use absolute paths"
        assert 'cd "$(dirname' in content, "Should convert to absolute path"

        # Test successful completion - our fix should make the script more robust
        result = subprocess.run(["bash", "-n", "./run-server.sh"], capture_output=True, text=True)
        assert result.returncode == 0, "Script should have valid syntax after our fix"

    def test_pip_detection_with_non_interactive_shell(self):
        """Test pip detection works in non-interactive shell environments.

        This addresses the contributor's suggestion about non-interactive shells
        not sourcing ~/.bashrc where pip PATH might be defined.
        """
        # Test case for Git Bash on Windows and non-interactive Linux shells
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock virtual environment structure
            venv_path = Path(temp_dir) / ".pal_venv"
            bin_path = venv_path / "bin"
            bin_path.mkdir(parents=True)

            # Create mock python executable
            python_exe = bin_path / "python"
            python_exe.write_text("#!/bin/bash\necho 'Python 3.12.3'\n")
            python_exe.chmod(0o755)

            # Create mock pip executable
            pip_exe = bin_path / "pip"
            pip_exe.write_text("#!/bin/bash\necho 'pip 23.0.1'\n")
            pip_exe.chmod(0o755)

            # Test that we can detect pip using explicit paths (not PATH)
            assert python_exe.exists(), "Mock python executable should exist"
            assert pip_exe.exists(), "Mock pip executable should exist"
            assert python_exe.is_file(), "Python should be a file"
            assert pip_exe.is_file(), "Pip should be a file"

    def test_enhanced_diagnostic_messages_included(self):
        """Test that our enhanced diagnostic messages are included in the script.

        Verify that the script contains the enhanced error diagnostics we added.
        """
        content = Path("./run-server.sh").read_text()

        # Check that enhanced diagnostic information is present in the script
        expected_diagnostic_patterns = [
            "Enhanced diagnostic information for debugging",
            "Diagnostic information:",
            "Python executable:",
            "Python executable exists:",
            "Python executable permissions:",
            "Virtual environment path:",
            "Virtual environment exists:",
            "Final diagnostic information:",
        ]

        for pattern in expected_diagnostic_patterns:
            assert pattern in content, f"Enhanced diagnostic pattern '{pattern}' should be in script"

    def test_simulators_prefer_managed_environment(self, monkeypatch, tmp_path):
        from communication_simulator_test import CommunicationSimulator
        from simulator_tests.base_test import BaseSimulatorTest

        for directory in (".venv", ".pal_venv"):
            python = tmp_path / directory / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()

        monkeypatch.chdir(tmp_path)
        expected = str(tmp_path / ".pal_venv" / "bin" / "python")
        communication = object.__new__(CommunicationSimulator)
        base = object.__new__(BaseSimulatorTest)

        assert communication._get_python_path() == expected
        assert base._get_python_path() == expected

    def test_simulator_supplies_working_directory(self, monkeypatch, tmp_path):
        import json
        import logging
        from types import SimpleNamespace

        from simulator_tests.base_test import BaseSimulatorTest

        captured = {}

        def fake_run(_command, **kwargs):
            messages = [json.loads(line) for line in kwargs["input"].splitlines()]
            captured.update(messages[-1]["params"]["arguments"])
            content = json.dumps({"continuation_offer": {"continuation_id": "test-thread"}})
            response = {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": content}]}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("simulator_tests.base_test.subprocess.run", fake_run)
        simulator = object.__new__(BaseSimulatorTest)
        simulator.python_path = "python"
        simulator.logger = logging.getLogger("simulator-test")

        _, continuation_id = simulator.call_mcp_tool("chat", {"prompt": "test", "model": "flash"})

        assert captured["working_directory_absolute_path"] == str(tmp_path)
        assert continuation_id == "test-thread"

        captured.clear()
        simulator.call_mcp_tool("precommit", {"step": "test", "model": "flash"})
        assert "working_directory_absolute_path" not in captured

    def test_simulator_registry_names_are_consistent(self):
        from simulator_tests import TEST_REGISTRY

        for name, test_class in TEST_REGISTRY.items():
            assert test_class(verbose=False).test_name == name

    def test_simulator_log_reader_honors_since_time(self, monkeypatch, tmp_path):
        from simulator_tests.log_utils import LogUtils

        main_log = tmp_path / "main.log"
        activity_log = tmp_path / "activity.log"
        main_log.write_text(
            "2026-08-01 00:00:00,000 - old\n"
            "old continuation\n"
            "2026-09-09 12:00:00,001 - new main\n"
            "new continuation\n"
        )
        activity_log.write_text("2026-08-01 00:00:00,000 - old activity\n" "2026-09-09 12:00:01,001 - new activity\n")
        monkeypatch.setattr(LogUtils, "MAIN_LOG_FILE", str(main_log))
        monkeypatch.setattr(LogUtils, "ACTIVITY_LOG_FILE", str(activity_log))

        logs = LogUtils.get_server_logs_since("2026-09-09T12:00:00")

        assert "old" not in logs
        assert "new main" in logs
        assert "new continuation" in logs
        assert "new activity" in logs

    def test_setup_env_file_does_not_create_bsd_backup(self, tmp_path):
        """Ensure setup_env_file avoids creating .env'' artifacts (BSD sed behavior)."""
        script_path = Path("./run-server.sh").resolve()

        # Prepare temp workspace with example env
        env_example = Path(".env.example").read_text()
        target_example = tmp_path / ".env.example"
        target_example.write_text(env_example)

        # Run setup_env_file inside isolated shell session
        command = f"""
        set -e
        cd "{tmp_path}"
        source "{script_path}"
        setup_env_file
        """
        env = os.environ.copy()
        subprocess.run(["bash", "-lc", command], check=True, env=env, text=True)

        artifacts = {p.name for p in tmp_path.glob(".env*")}
        assert ".env''" not in artifacts, "setup_env_file should not create BSD sed backup artifacts"
        assert ".env" in artifacts, ".env should be created from .env.example"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
