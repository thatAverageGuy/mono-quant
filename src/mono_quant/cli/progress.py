"""Progress bar utilities for CLI.

This module provides CI/CD-friendly progress bar wrappers using tqdm.
Progress bars are automatically disabled in CI environments (when CI env var is set)
or when output is not a TTY (e.g., when redirected to a file).

## Functions

- **should_show_progress**: Detect if progress bars should be shown
- **cli_progress**: Wrapper around tqdm with CI/TTY detection

Example:
    >>> from mono_quant.cli.progress import cli_progress
    >>> for item in cli_progress(range(100), desc="Processing"):
    ...     process(item)
"""

import os
import sys
from typing import Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def should_show_progress() -> bool:
    """Determine if progress bars should be displayed.

    Progress bars are disabled in CI/CD environments or when stdout is not a TTY.
    This ensures clean output in automated systems while providing visual feedback
    for interactive terminal sessions.

    Returns:
        True if progress should be shown, False otherwise.

    Detection logic:
        - If CI environment variable is set: return False
        - If stdout is not a TTY: return False
        - Otherwise: return True
    """
    # Check for CI environment variable (set by GitHub Actions, Travis CI, etc.)
    if os.getenv("CI"):
        return False

    # Check if stdout is a TTY (terminal)
    # hasattr check for compatibility with different environments
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return True

    return False


def cli_progress(
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    **kwargs,
) -> tqdm:
    """Create a tqdm progress bar with CI/TTY-aware auto-disable.

    This wrapper automatically disables progress bars in CI environments or
    when stdout is not a TTY. Use this for all CLI progress displays.

    Args:
        iterable: The iterable to track progress over
        desc: Description text to show next to the progress bar
        total: Total number of items (auto-detected if None)
        **kwargs: Additional arguments passed to tqdm

    Returns:
        A tqdm iterable that yields items from the input iterable

    Example:
        >>> items = list(range(100))
        >>> for item in cli_progress(items, desc="Quantizing"):
        ...     # Process item
        ...     pass
    """
    return tqdm(
        iterable,
        desc=desc,
        disable=not should_show_progress(),
        total=total,
        **kwargs,
    )


__all__ = [
    "should_show_progress",
    "cli_progress",
]
