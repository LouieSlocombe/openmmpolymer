"""Invalid CLI requests fail before building, reading inputs or writing output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openmmpolymer import __main__ as cli

CRYSTAL = "--protocol tm --crystal-pdb crystal.pdb --system-xml system.xml"
YOUNGS_RATE = "--modulus-relax-times 10,50,100 --target-strain-rate 0.01"


@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        # Every protocol validates its full budget, including during a dry run.
        *(
            (
                CRYSTAL if name == "tm" else f"[*]CC[*] --protocol {name}",
                "--dry-run --max-total-ns 0.001",
            )
            for name in sorted(cli.PROTOCOLS)
        ),
        *(
            (f"[*]CC[*] --protocol {name} --dry-run", flags)
            for name, flags in (
                ("melt-quench", "--step-k 0"),
                ("melt-quench", "--t-end 700"),
                ("tg", "--fine-step-k 0"),
                ("modulus", "--strain-increment 0"),
                ("modulus", "--elastic-strain-limit 0.5"),
                ("relax", "--step-strain 0"),
                ("equilibrate", "-r TOOLONG"),
            )
        ),
        # Tensile criteria and ladders are checked before chemistry begins.
        *(
            (f"[*]CC[*] --protocol {name}", flags)
            for name, flags in (
                ("breaking", "--failure-fraction 1.5"),
                ("elongation", "--failure-fraction 1.5"),
                *(
                    (name, f"--{name}-strain-increment 0")
                    for name in ("breaking", "elongation", "yield")
                ),
                ("elongation", "--confirmation-steps 0"),
                ("elongation", "--elongation-replicas 0"),
                ("yield", "--yield-offset-strain 0"),
                ("yield", "--yield-fit-min-strain 0.03 --yield-fit-max-strain 0.02"),
                ("yield", "--yield-fit-max-strain 0.5"),
            )
        ),
        # Generic rate controls: missing, incompatible and conflicting options.
        *(
            ("[*]CC[*] --protocol yield --dry-run", flags)
            for flags in (
                "--rate-property yield_strength --rate-hold-times 1,2,3",
                "--rate-property bulk_modulus --rate-hold-times 1,2,3 "
                "--target-strain-rate 0.1",
                "--rate-property yield_strength --rate-hold-times 1,1,2 "
                "--target-property-rate 0.1",
                *(
                    "--rate-property yield_strength --rate-hold-times 1,2,3 "
                    f"--target-property-rate {target}"
                    for target in ("nan", "-1")
                ),
                *(
                    "--rate-property yield_strength --rate-hold-times 1,2,3 "
                    f"--target-property-rate 0.1 {extra}"
                    for extra in (
                        "--max-total-ns .001",
                        "--max-rate-extrapolation-decades -1",
                        "--target-strain-rate 0.2",
                        "--modulus-relax-times 1,2,4",
                    )
                ),
                "--rate-property yield_strength --modulus-relax-times 1,2,3 "
                "--target-property-rate 0.1",
            )
        ),
        # The original Young's-modulus aliases retain the same validation.
        *(
            ("[*]CC[*] --protocol modulus", flags)
            for flags in (
                "--modulus-relax-times 10,50,100",
                "--target-strain-rate 0.01",
                *(
                    f"--modulus-relax-times {holds} --target-strain-rate 0.01"
                    for holds in ("10,50", "10,10,50", "10,nan,50", "10,-50,100")
                ),
                *(
                    f"--modulus-relax-times 10,50,100 --target-strain-rate {target}"
                    for target in ("0", "inf")
                ),
                f"{YOUNGS_RATE} --max-total-ns 0.001 --dry-run",
            )
        ),
        ("[*]CC[*] --protocol yield", YOUNGS_RATE),
        ("--analyse .", YOUNGS_RATE),
        # Convergence diagnostics only accept one saved run and valid windows.
        *(
            ("--convergence", flags)
            for flags in (
                "",
                "--analyse one two",
                "--analyse one --target-strain-rate 0.1",
                "--analyse one --rate-property yield_strength --target-property-rate 0.1",
                "--analyse one --window-fractions .5,1",
                "--analyse one --convergence-tolerance nan",
                "--analyse one --min-effective-samples 0",
                "--analyse one --convergence-discard-fraction 1",
            )
        ),
        # Melting needs a prepared crystal and a usable heating schedule.
        ("--protocol tm", ""),
        ("--protocol tm --crystal-pdb crystal.pdb", ""),
        (f"[*]CC[*] {CRYSTAL}", ""),
        ("[*]CC[*] --crystal-pdb crystal.pdb", ""),
        *(
            (CRYSTAL, flags)
            for flags in (
                "--t-start 500 --t-end 300",
                "--step-k 0",
                "--hold-ps -1",
                "--max-total-ns 0.001",
                "--check-melt",
            )
        ),
    ],
)
def test_invalid_request_has_no_side_effects(
    mode: str,
    flags: str,
    no_build: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("an invalid request read the crystal")

    monkeypatch.setattr(cli, "load_crystal", unexpected_read)
    with pytest.raises(SystemExit, match="2"):
        cli.main([*mode.split(), *flags.split(), "-o", "output"])
    assert not no_build
    assert not Path("output").exists()
