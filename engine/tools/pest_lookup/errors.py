"""Exception classes for the pest_lookup module.

Mirrors engine/tools/agri_calc/errors.py so agent error-boundary handling
is consistent across tools.
"""

class PestLookupError(Exception):
    """Base exception class for all pest_lookup errors."""
    pass

class InvalidInputError(PestLookupError):
    """Raised when the input pest name doesn't match any known pest
    keyword at all (e.g. a typo, or a pest category outside the
    vocabulary this corpus was extracted against). This is a caller
    error, distinct from a valid pest with zero corpus coverage."""
    pass
