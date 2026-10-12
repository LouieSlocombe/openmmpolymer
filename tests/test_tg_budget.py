"""The Tg quote must cover the ladder actually selected after coarse cooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import tg
from openmmpolymer._workflow import spec_request
from openmmpolymer.protocols import Protocol, RunManifest, RunSummary, _run_identity
from openmmpolymer.simulate import RunContext

from .helpers import forbidden


class FineDynamicsReached(RuntimeError):
    """Stop at the fine dynamics boundary, after all budget checks."""


def _stub_coarse_dynamics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    surviving_waypoints: tuple[float, ...],
) -> list[Protocol]:
    """Write coarse state records, keeping real restart and duration arithmetic."""
    fine_calls: list[Protocol] = []

    def run_protocol(
        protocol: Protocol,
        run: RunContext,
        directory: Path,
        **options: Any,
    ) -> RunSummary:
        if any(stage.name.startswith(tg.FINE_STEM) for stage in protocol.stages):
            fine_calls.append(protocol)
            raise FineDynamicsReached
        manifest = RunManifest.load(directory) or RunManifest(
            protocol=protocol.name,
            seed=run.seed,
            provenance={"version": 1, "run": _run_identity(run), "stages": {}},
        )
        for stage in protocol.stages:
            state = directory / f"{stage.name}.xml"
            state.write_text("synthetic saved state")
            recorded: dict[str, Any] = {"final_state": str(state)}
            if stage.kind == "quench":
                temperatures = stage.options["temperatures_k"]
                waypoints = []
                for temperature in temperatures:
                    waypoint = directory / f"waypoint_{temperature:g}.xml"
                    if temperature in surviving_waypoints:
                        waypoint.write_text("synthetic coarse waypoint")
                    waypoints.append(str(waypoint))
                recorded.update(
                    samples={"segment_temperature_k": temperatures},
                    waypoints=waypoints,
                )
            manifest.stages[stage.name] = recorded
        manifest.save(directory)
        return RunSummary(
            protocol=protocol.name,
            run_dir=str(directory),
            manifest_path=str(directory / "manifest.json"),
            results=(),
            skipped=(),
            wall_seconds=0.0,
            final_state=str(state),
        )

    monkeypatch.setattr(tg, "run_protocol", run_protocol)
    # The requested transition is explicit; density fitting cannot affect these
    # tests of the restart state and the time charged to that restart.
    monkeypatch.setattr(tg, "quench_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(tg, "_fit_coarse", lambda *args: (None, "synthetic coarse"))
    return fine_calls


def test_default_quote_covers_the_actual_420_k_restart_ladder() -> None:
    spec = tg.TgSpec()
    top, bottom = tg.fine_window(420.0, spec)
    chosen = tg.pick_waypoint(
        [
            (temperature, "state")
            for temperature in tg.coarse_schedule(spec).temperatures_k
        ],
        top,
    )
    assert chosen == (500.0, "state")
    actual = tg.fine_schedule(chosen[0], bottom, spec)
    assert actual.n_temperatures == 29
    coarse = tg.tg_coarse_scan(spec)
    assert (coarse.total_duration_ps + actual.total_ps) / 1000 == 112.11
    assert tg.nominal_fine_schedule(spec).n_temperatures == 30
    assert tg.tg_scan_duration_ps(spec) / 1000 == 115.11


@pytest.mark.parametrize(
    "spec",
    [
        tg.TgSpec(),
        tg.TgSpec(coarse_step_k=33.3, fine_step_k=7.3, window_k=63.0),
        tg.TgSpec(window_k=300.0),
    ],
)
def test_quote_bounds_intact_restart_ladders_throughout_the_coarse_range(
    spec: tg.TgSpec,
) -> None:
    coarse = tg.coarse_schedule(spec)
    candidates = [(temperature, "state") for temperature in coarse.temperatures_k]
    quoted = tg.nominal_fine_schedule(spec)
    for temperature in coarse.temperatures_k:
        for offset in (-0.01, 0.0, 0.01):
            top, bottom = tg.fine_window(temperature + offset, spec)
            chosen = tg.pick_waypoint(candidates, top)
            assert chosen is not None
            actual = tg.fine_schedule(chosen[0], bottom, spec)
            assert actual.total_ps <= quoted.total_ps
    assert (
        quoted.n_temperatures
        <= tg.fine_schedule(
            spec.melt_temperature_k, spec.t_floor_k, spec
        ).n_temperatures
    )


def test_105_ns_budget_refuses_default_scan_before_writes_or_dynamics(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = Path("run")
    monkeypatch.setattr(tg, "run_protocol", forbidden("Over-budget scan ran dynamics"))
    with pytest.raises(tg.TgError, match=r"max_total_ns|budget"):
        tg.run_tg_scan(
            argon_run,
            directory,
            spec=tg.TgSpec(max_total_ns=105.0),
            tg_approx_k=420.0,
        )
    assert not directory.exists()


@pytest.mark.parametrize(
    ("surviving_waypoints", "transition_k", "budget_ns", "actual_ns"),
    [
        ((650.0, 550.0), 420.0, 120.0, 142.11),
        ((), 210.0, 116.0, 117.11),
    ],
    ids=["hotter-surviving-waypoint", "fallback-precool"],
)
def test_resolved_restart_is_checked_before_any_fine_dynamics(
    surviving_waypoints: tuple[float, ...],
    transition_k: float,
    budget_ns: float,
    actual_ns: float,
    argon_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine_calls = _stub_coarse_dynamics(
        monkeypatch, surviving_waypoints=surviving_waypoints
    )
    spec = tg.TgSpec(max_total_ns=budget_ns)
    assert tg.tg_scan_duration_ps(spec) / 1000 < budget_ns < actual_ns
    with pytest.raises(tg.TgError, match=r"max_total_ns|budget"):
        tg.run_tg_scan(argon_run, "run", spec=spec, tg_approx_k=transition_k)
    assert not fine_calls
    assert Path("run/manifest.json").is_file()


def test_every_rate_is_budgeted_before_the_first_fine_pass(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fine_calls = _stub_coarse_dynamics(monkeypatch, surviving_waypoints=(650.0, 550.0))
    spec = tg.TgSpec(max_total_ns=160.0)
    rates = (10.0, 5.0, 2.0)
    assert tg.tg_scan_duration_ps(spec, rates_k_per_ns=rates) / 1000 == 145.11
    coarse_ps = tg.tg_coarse_scan(spec).total_duration_ps
    passes_ps = [
        tg.fine_schedule(550.0, 360.0, spec, hold_ps=5000.0 / rate).total_ps
        for rate in rates
    ]
    assert all((coarse_ps + duration) / 1000 < 160.0 for duration in passes_ps)
    assert (coarse_ps + sum(passes_ps)) / 1000 == 181.11
    with pytest.raises(tg.TgError, match=r"max_total_ns|budget"):
        tg.cooling_rate_series(
            argon_run, "run", spec=spec, rates_k_per_ns=rates, tg_approx_k=420.0
        )
    assert not fine_calls


def test_shared_fallback_precool_is_charged_only_once_across_rates(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fine_calls = _stub_coarse_dynamics(monkeypatch, surviving_waypoints=())
    # 25.11 ns preparation + 100 ns fine passes + one 17 ns pre-cool fits;
    # charging the same pre-cool once per rate would exceed this budget.
    with pytest.raises(FineDynamicsReached):
        tg.cooling_rate_series(
            argon_run,
            "run",
            spec=tg.TgSpec(max_total_ns=145.11),
            rates_k_per_ns=(10.0, 5.0, 2.0),
            tg_approx_k=210.0,
        )
    assert len(fine_calls) == 1
    assert fine_calls[0].stages[0].name == tg.PRECOOL_STEM


def test_resume_budget_includes_the_precool_already_recorded(
    argon_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = Path("run")
    directory.mkdir()
    spec = tg.TgSpec(max_total_ns=116.0)
    fine_state = directory / "fine.xml"
    fine_state.write_text("synthetic completed fine chunk")
    precool_state = directory / "precool.xml"
    precool_state.write_text("synthetic completed pre-cool")
    manifest = RunManifest(
        protocol=tg.PROTOCOL_NAME,
        seed=argon_run.seed,
        provenance={"version": 1, "run": _run_identity(argon_run), "stages": {}},
        stages={
            tg.PRECOOL_STEM: {
                "final_state": str(precool_state),
                "duration_ps": 17_000.0,
                "samples": {
                    "segment_temperature_k": [*range(650, 274, -25), 270],
                    "segment_duration_ps": [1000.0] * 17,
                },
            },
            f"{tg.FINE_STEM}_00": {"final_state": str(fine_state)},
        },
    )
    manifest.save(directory)
    (directory / tg.WORKFLOW_NAME).write_text(
        json.dumps(
            {
                "request": spec_request(spec, tg_approx_k=210.0),
                "restart": "precool",
                "start_state": str(directory / "05_npt.xml"),
                "start_temperature_k": 270.0,
            }
        )
    )
    fine_calls = _stub_coarse_dynamics(monkeypatch, surviving_waypoints=())
    # The fine pass is already under way, so restart selection will not return
    # a new pre-cool stage. Its 17 ns still belongs to this scan's total cost.
    with pytest.raises(tg.TgError, match=r"max_total_ns|budget"):
        tg.run_tg_scan(argon_run, directory, spec=spec, tg_approx_k=210.0)
    assert not fine_calls
