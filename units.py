from numpy import deg2rad as deg2rad


def mph2mps(x: float) -> float:
    """Convert miles per hour to meters per second."""
    
    return x * 0.44704


def m2y(x: float) -> float:
    """Convert meters to yards."""
    
    return x * 1.09361


def rpm2radps(x: float) -> float:
    """Convert rotations per minute to radians per second."""

    return deg2rad(x / 60 * 360)


def s2lr(x: str) -> float:
    """Mark a signed offline measurement with L or R.
    
    For example, -0.5 becomes L0.5 and 300 becomes R300. We use the
    x-axis as forward, the y-axis as upward, and z-axis as right.
    """
    
    return "L" if x < 0 else "R" if x > 0 else ""


def lr2s(x: str) -> float:
    """Convert a offline measurement to signed.
    
    For example, L0.5 becomes -0.5 and R300 becomes 300. We use the
    x-axis as forward, the y-axis as upward, and z-axis as right.
    """
    
    if x.startswith("L"):
        return -float(x[1:])
    elif x.startswith("R"):
        return float(x[1:])
    return float(x)