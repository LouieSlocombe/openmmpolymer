"""Mechanical and tensile measurements must describe the same deformation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from openmmpolymer.mechanical import (
    ModulusSchedule,
    ModulusSpec,
    deform_protocol,
    deform_schedule,
)
from openmmpolymer.tensile import (
    BreakingSpec,
    ElongationSpec,
    TensileSchedule,
    TensileSpec,
    YieldSpec,
    tensile_protocol,
    tensile_schedule,
)


@pytest.mark.parametrize(
    ("spec_type", "name"),
    [(BreakingSpec, "breaking"), (ElongationSpec, "elongation"), (YieldSpec, "yield")],
)
@pytest.mark.parametrize(
    ("stage_ps", "holds", "starts"),
    [
        (25.0, [2, 2, 1], [0.0, 0.21, 0.4641]),
        (10.0, [1, 1, 1, 1, 1], [0.0, 0.1, 0.21, 0.331, 0.4641]),
        (100.0, [5], [0.0]),
    ],
)
def test_matched_workflows_preserve_the_same_ladder_across_chunks(
    spec_type: type[TensileSpec],
    name: str,
    stage_ps: float,
    holds: list[int],
    starts: list[float],
) -> None:
    """Five 10% increments reach 61.051%, regardless of where a run resumes."""
    mechanical = ModulusSpec(
        temperature_k=310.0,
        pressure_bar=1.7,
        axis=1,
        strain_increment=0.1,
        max_strain=0.5,
        relax_ps=10.0,
        samples_per_step=12,
        stage_ps=stage_ps,
    )
    tensile = spec_type(
        temperature_k=mechanical.temperature_k,
        pressure_bar=mechanical.pressure_bar,
        axis=mechanical.axis,
        strain_increment=mechanical.strain_increment,
        max_strain=mechanical.max_strain,
        relax_ps=mechanical.relax_ps,
        samples_per_step=mechanical.samples_per_step,
        stage_ps=stage_ps,
    )
    reference = (3.2, 4.3, 5.4)
    modulus_ladder = deform_protocol(
        mechanical, timestep_fs=1.0, replica=7, reference_box_nm=reference
    )
    tensile_ladder = tensile_protocol(
        tensile, timestep_fs=1.0, replica=7, reference_box_nm=reference
    )

    assert modulus_ladder.name == "mechanical"
    assert tensile_ladder.name == name
    assert [stage.name for stage in modulus_ladder.stages] == [
        f"06_deform_r7_{index:02d}" for index in range(len(holds))
    ]
    assert [stage.name for stage in tensile_ladder.stages] == [
        f"06_{name}_r7_{index:03d}" for index in range(len(holds))
    ]
    assert [stage.options for stage in modulus_ladder.stages] == [
        stage.options for stage in tensile_ladder.stages
    ]
    assert modulus_ladder.stages[0].options == {
        "temperature_k": 310.0,
        "pressure_bar": 1.7,
        "axis": 1,
        "strain_increment": 0.1,
        "n_steps": holds[0],
        "relax_ps": 10.0,
        "strain_start": 0.0,
        "samples_per_step": 12,
        "timestep_fs": 1.0,
        "new_velocities": True,
        "reference_box_nm": list(reference),
    }
    for ladder in (modulus_ladder, tensile_ladder):
        assert all(stage.kind == "deform" for stage in ladder.stages)
        assert [stage.options["n_steps"] for stage in ladder.stages] == holds
        assert [stage.options["strain_start"] for stage in ladder.stages] == (
            pytest.approx(starts)
        )
        assert [stage.options["new_velocities"] for stage in ladder.stages] == [
            True,
            *([False] * (len(holds) - 1)),
        ]
        assert all(
            stage.options["reference_box_nm"] == list(reference)
            for stage in ladder.stages
        )
        ends = [
            (1.0 + stage.options["strain_start"])
            * (1.0 + stage.options["strain_increment"]) ** stage.options["n_steps"]
            - 1.0
            for stage in ladder.stages
        ]
        assert ends[:-1] == pytest.approx(starts[1:])
        assert ends[-1] == pytest.approx(0.61051)
        assert [stage.duration_ps for stage in ladder.stages] == [
            count * 10.0 for count in holds
        ]
        assert ladder.total_duration_ps == 50.0

    modulus_schedule = deform_schedule(mechanical)
    tensile_result = tensile_schedule(tensile)
    assert type(modulus_schedule) is ModulusSchedule
    assert type(tensile_result) is TensileSchedule
    assert (
        asdict(modulus_schedule)
        == asdict(tensile_result)
        == {
            "n_steps": 5,
            "increment": 0.1,
            "relax_ps": 10.0,
        }
    )
    for schedule in (modulus_schedule, tensile_result):
        assert schedule.max_strain == pytest.approx(0.61051)
        assert schedule.total_ps == 50.0
        assert schedule.strain_rate_per_ns == pytest.approx(12.2102)


@pytest.mark.parametrize("spec_type", [BreakingSpec, ElongationSpec, YieldSpec])
def test_optional_recording_is_added_to_every_tensile_chunk(
    spec_type: type[TensileSpec],
) -> None:
    spec = spec_type(strain_increment=0.1, max_strain=0.5, relax_ps=10.0, stage_ps=25.0)
    plain = tensile_protocol(spec)
    recorded = tensile_protocol(replace(spec, trajectory_ps=2.5))
    assert recorded.total_duration_ps == plain.total_duration_ps == 50.0
    for original, recording in zip(plain.stages, recorded.stages, strict=True):
        assert original.name == recording.name
        assert "reference_box_nm" not in recording.options
        assert "trajectory" not in original.options
        trajectory = recording.options["trajectory"]
        assert trajectory.format == "xtc"
        assert trajectory.interval_ps == 2.5
        assert {
            key: value
            for key, value in recording.options.items()
            if key != "trajectory"
        } == original.options


@pytest.mark.parametrize("schedule_type", [ModulusSchedule, TensileSchedule])
def test_public_schedule_dataclasses_keep_their_constructor_and_fields(
    schedule_type: type[ModulusSchedule] | type[TensileSchedule],
) -> None:
    schedule = schedule_type(5, 0.1, 10.0)
    assert schedule == schedule_type(n_steps=5, increment=0.1, relax_ps=10.0)
    assert asdict(schedule) == {"n_steps": 5, "increment": 0.1, "relax_ps": 10.0}
    assert replace(schedule, relax_ps=20.0).total_ps == 100.0
    with pytest.raises(FrozenInstanceError):
        schedule.n_steps = 6  # type: ignore[misc]


def test_tensile_chunks_copy_the_callers_reference_box_independently() -> None:
    reference = [4, 5, 6]
    spec = BreakingSpec(
        strain_increment=0.1, max_strain=0.5, relax_ps=10.0, stage_ps=25.0
    )
    ladder = tensile_protocol(spec, reference_box_nm=reference)
    boxes = [stage.options["reference_box_nm"] for stage in ladder.stages]
    assert all(box == [4, 5, 6] for box in boxes)
    assert all(type(value) is int for box in boxes for value in box)

    reference[0] = 7
    assert all(box == [4, 5, 6] for box in boxes)
    boxes[0][1] = 8
    assert reference == [7, 5, 6]
    assert boxes[1:] == [[4, 5, 6], [4, 5, 6]]
