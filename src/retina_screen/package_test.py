"""Package-level smoke test: import succeeds, public surface is intentional."""

import retina_screen


def test_package_imports() -> None:
    assert retina_screen.__all__ == []
