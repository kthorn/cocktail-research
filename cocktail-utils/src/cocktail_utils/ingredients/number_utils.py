from decimal import Decimal


def _is_integer(text: str) -> bool:
    """Check if a string represents a valid integer."""
    try:
        int(text)
        return True
    except ValueError:
        return False


def _is_number(text: str) -> bool:
    """Check if a string represents a valid number (int or float)."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_fraction(text: str) -> bool:
    """Check if a string represents a valid fraction (e.g., '1/2')."""
    if "/" not in text:
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(_is_integer(part) for part in parts)


def _parse_fraction(text: str) -> Decimal:
    """Parse a fraction string (e.g., '1/2') into a Decimal."""
    if "/" not in text:
        raise ValueError(f"Not a fraction: {text}")

    numerator_str, denominator_str = text.split("/")
    numerator = Decimal(numerator_str)
    denominator = Decimal(denominator_str)

    if denominator == 0:
        raise ZeroDivisionError("Division by zero in fraction")

    return numerator / denominator
