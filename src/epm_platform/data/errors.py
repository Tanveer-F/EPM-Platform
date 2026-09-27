"""Sanitized data-pipeline failures."""


class DataError(ValueError):
    """An actionable error safe to print without credentials or signed URLs."""
