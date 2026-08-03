"""Exception classes for the crop_calendar module.

Mirrors engine/tools/agri_calc/errors.py so agent error-boundary handling
is consistent across tools.
"""

class CropCalendarError(Exception):
    """Base exception class for all crop_calendar errors."""
    pass

class InvalidInputError(CropCalendarError):
    """Raised when user-supplied inputs are invalid (e.g., unknown crop name)."""
    pass
