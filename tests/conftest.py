"""Pytest-wide test taxonomy for the offline suite."""

from __future__ import annotations

from pathlib import Path

import pytest


INTEGRATION_MODULES = {
    "test_graph_topology.py",
    "test_import_approval.py",
    "test_search_approval.py",
    "test_system.py",
    "test_workflow_contracts.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify legacy tests without requiring marker boilerplate per file."""
    for item in items:
        existing_markers = {marker.name for marker in item.iter_markers()}
        if "live" in existing_markers:
            continue
        module_name = Path(str(item.fspath)).name
        if "integration" in existing_markers or module_name in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
