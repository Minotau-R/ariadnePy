from __future__ import annotations


class AriadneError(Exception):
    """Base exception for all ariadnepy errors."""


class AriadneDownloadError(AriadneError):
    """Raised when a network download fails."""


class AriadneVersionError(AriadneError):
    """Raised when a requested resource version is not available."""


class AriadneParseError(AriadneError):
    """Raised when a downloaded file cannot be parsed."""


class AriadnePathError(AriadneError):
    """Raised when no valid path exists between two graph nodes."""


class AriadneCacheError(AriadneError):
    """Raised when the local cache cannot be read or written."""
