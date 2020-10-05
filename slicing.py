
def shift_slice(original,steps):
    """
    Shifts a slice _step_ steps.
    """
    noneadd = lambda x,y: None if x is None or y is None else x+y
    zfloor = lambda x: 0 if x < 0 else x

    try:
        shifted = noneadd(original, steps)
    except TypeError:
        shifted = slice(
            zfloor(noneadd(original.start, steps)),
            zfloor(noneadd(original.stop, steps)),
            original.step
        )

    return shifted
