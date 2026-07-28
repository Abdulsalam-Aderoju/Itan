"""Exception classes for the agri_calc module.

These exception classes distinguish between validation errors and database missing
constants to allow robust agent error boundary handling.
"""

class AgriCalcError(Exception):
    """Base exception class for all agri_calc errors."""
    pass

class InvalidInputError(AgriCalcError):
    """Raised when user-supplied inputs are invalid (e.g., negative or zero values)."""
    pass

class MissingConstantError(AgriCalcError):
    """Raised when the database lacks the required agronomic constant for a lookup."""
    pass
