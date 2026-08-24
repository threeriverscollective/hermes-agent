"""Tests for cgroup resource-limit gating in the docker backend.

On hosts where the cgroup v2 cpu/memory/pids controllers are not delegated
(e.g. unprivileged Proxmox LXCs), passing ``--cpus``/``--memory``/``--pids-limit``
to ``docker run`` fails every container start with OCI runtime error / exit 126.
``_cgroup_limits_available`` probes once and the resource flags are gated on it,
so the sandbox degrades gracefully instead of failing.
"""
import subprocess

import pytest

import tools.environments.docker as docker_env


@pytest.fixture(autouse=True)
def _reset_cgroup_cache():
    """The probe result is cached in a module-level global; reset per test."""
    docker_env._cgroup_limits_ok = None
    yield
    docker_env._cgroup_limits_ok = None


def test_pids_limit_not_in_base_security_args():
    """``--pids-limit`` must NOT be hardcoded in the static security args.

    It requires the pids cgroup controller and is gated on the probe instead.
    """
    assert "--pids-limit" not in docker_env._BASE_SECURITY_ARGS


def test_probe_returns_true_when_container_starts(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    captured = {}

    def _run(cmd, *a, **k):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    assert docker_env._cgroup_limits_available("hermes-agent:latest") is True
    # Probes all three controllers together against the real sandbox image.
    assert "--cpus" in captured["cmd"]
    assert "--memory" in captured["cmd"]
    assert "--pids-limit" in captured["cmd"]
    assert "hermes-agent:latest" in captured["cmd"]


def test_probe_result_is_cached(monkeypatch):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env._cgroup_limits_available("img")
    docker_env._cgroup_limits_available("img")
    docker_env._cgroup_limits_available("img")
    assert len(calls) == 1  # probe runs once, then cached


def _required_mode_harness(monkeypatch, *, run_result=None, run_error=None):
    """Construct one required-mode Docker environment without a Docker daemon."""
    commands = []

    monkeypatch.setattr(docker_env, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_image_uses_init_entrypoint", lambda *args: False)
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(docker_env, "_egress_proxy_args_for_docker", lambda: ([], {}, []))
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)

    def _run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        if len(cmd) > 1 and cmd[1] == "run":
            if run_error is not None:
                raise run_error
            return run_result or subprocess.CompletedProcess(
                cmd, 0, stdout="worker-container-id\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = docker_env.DockerEnvironment(
        image="worker-image",
        cwd="/workspace",
        timeout=60,
        cpu=2,
        memory=4096,
        disk=0,
        task_id="required-mode-test",
        network=False,
        persist_across_processes=False,
        docker_require_resource_limits=True,
        extra_args=[
            "--read-only",
            "--pull=never",
            "--label",
            "hermes-run-id=run:real-model",
        ],
    )
    return env, commands


def test_required_mode_skips_probe_and_puts_exact_limits_on_worker_run(monkeypatch):
    def _probe_must_not_run(_image):
        pytest.fail("required mode must not invoke the cgroup capability probe")

    monkeypatch.setattr(docker_env, "_cgroup_limits_available", _probe_must_not_run)
    _env, commands = _required_mode_harness(monkeypatch)

    run_commands = [cmd for cmd in commands if cmd[1:3] == ["run", "-d"]]
    assert len(run_commands) == 1
    worker_run = run_commands[0]
    assert worker_run[worker_run.index("--cpus") : worker_run.index("--cpus") + 2] == [
        "--cpus",
        "2",
    ]
    assert worker_run[worker_run.index("--memory") : worker_run.index("--memory") + 2] == [
        "--memory",
        "4096m",
    ]
    assert worker_run[
        worker_run.index("--pids-limit") : worker_run.index("--pids-limit") + 2
    ] == ["--pids-limit", docker_env._DEFAULT_PIDS_LIMIT]
    assert "--network=none" in worker_run
    assert "--read-only" in worker_run
    assert "--pull=never" in worker_run
    owner_label_index = worker_run.index("hermes-run-id=run:real-model")
    assert worker_run[owner_label_index - 1 : owner_label_index + 1] == [
        "--label",
        "hermes-run-id=run:real-model",
    ]


def test_required_mode_run_refusal_propagates_without_retry(monkeypatch):
    monkeypatch.setattr(
        docker_env,
        "_cgroup_limits_available",
        lambda _image: pytest.fail("required mode must not invoke the cgroup capability probe"),
    )
    run_error = subprocess.CalledProcessError(
        125, ["/usr/bin/docker", "run"], stderr="resource limit refused"
    )

    commands = []
    monkeypatch.setattr(docker_env, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_image_uses_init_entrypoint", lambda *args: False)
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(docker_env, "_egress_proxy_args_for_docker", lambda: ([], {}, []))
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)

    def _run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        if len(cmd) > 1 and cmd[1] == "run":
            raise run_error
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        docker_env.DockerEnvironment(
            image="worker-image",
            cwd="/workspace",
            cpu=2,
            memory=4096,
            disk=0,
            task_id="required-refusal-test",
            network=False,
            persist_across_processes=False,
            docker_require_resource_limits=True,
            extra_args=[
                "--read-only",
                "--pull=never",
                "--label",
                "hermes-run-id=run:real-model",
            ],
        )

    assert exc_info.value is run_error
    assert len([cmd for cmd in commands if cmd[1] == "run"]) == 1
    assert len([cmd for cmd in commands if cmd[1:3] == ["rm", "-f"]]) == 1
