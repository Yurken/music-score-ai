"""Custom exceptions for music-score-ai."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency required at runtime is missing."""


class InvalidInputError(ValueError):
    """Raised when user input file or parameters are invalid."""
