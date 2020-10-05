
def containedby(a,b):
    try:
        assert len(a.shape) == len(b.shape)
    except AssertionError:
        raise ValueError("Arrays must have the same number of dimensions")

    cb = True
    for ia,ib in zip(a.shape,b.shape):
        cb &= ia <= ib
    return cb
