"""Keep the README consistent with the files it shows and links to."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
FENCED_BLOCK = re.compile(r"^```(\w*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
# Markdown link and image targets, including nested badges, and HTML image sources.
LINK_TARGET = re.compile(r"\]\(\s*([^)\s]+)[^)]*\)|\bsrc=\"([^\"]+)\"")
URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def readme_section(heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)",
        README.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"README has no {heading!r} section"
    return match.group(1)


def test_readme_quickstart_is_the_quickstart_example():
    python_blocks = [
        body
        for language, body in FENCED_BLOCK.findall(readme_section("Quickstart"))
        if language == "python"
    ]
    assert python_blocks == [(ROOT / "examples" / "quickstart.py").read_text()]


def test_readme_relative_links_and_images_resolve_to_repository_files():
    prose = FENCED_BLOCK.sub("", README.read_text())
    targets = [markdown or html for markdown, html in LINK_TARGET.findall(prose)]
    relative = [
        target.split("#", 1)[0]
        for target in targets
        if not URL_SCHEME.match(target) and not target.startswith("#")
    ]
    assert relative, "README should link to repository files"
    missing = [
        target
        for target in relative
        if not (ROOT / target).resolve().is_relative_to(ROOT) or not (ROOT / target).is_file()
    ]
    assert not missing, f"README links to missing repository files: {missing}"


def test_example_catalogue_has_seven_single_species_and_one_neural_multispecies_script():
    expected = {
        "quickstart.py",
        "conductance.py",
        "pooled_regions.py",
        "mlpe_calibration.py",
        "region_transfer.py",
        "ensemble.py",
        "ibd.py",
        "multispecies.py",
    }
    assert {path.name for path in (ROOT / "examples").glob("*.py")} == expected
    linked = {
        Path(target).name
        for target, _ in LINK_TARGET.findall(readme_section("Examples"))
        if target.startswith("examples/")
    }
    assert linked == expected
