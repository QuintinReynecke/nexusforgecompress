import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def fast_delta_encode(data, residuals_dtype):
    """
    Accelerated Delta Encoding.
    """
    if not HAS_NUMBA:
        res = np.empty(data.shape, dtype=residuals_dtype)
        data_as_res = data.astype(residuals_dtype)
        res.ravel()[0] = data_as_res.ravel()[0]
        if data.size > 1:
            res.ravel()[1:] = np.diff(data_as_res.ravel())
        return res

    # Numba version
    # Cast to residuals_dtype FIRST to ensure math is done in target precision
    data_casted = data.ravel().astype(residuals_dtype)
    return _numba_delta_encode(data_casted).reshape(data.shape)

def fast_delta_decode(residuals, original_shape, original_dtype_name):
    """
    Accelerated Delta Decoding (Reconstruction).
    """
    if not HAS_NUMBA:
        if np.issubdtype(residuals.dtype, np.floating):
            cumsum_dtype = residuals.dtype
        else:
            cumsum_dtype = np.int64
        
        reconstructed = np.cumsum(residuals.astype(cumsum_dtype), dtype=cumsum_dtype).reshape(original_shape)
        return reconstructed.astype(original_dtype_name)

    # Numba version
    decoded = _numba_delta_decode(residuals.ravel())
    return decoded.astype(original_dtype_name).reshape(original_shape)

if HAS_NUMBA:
    @njit(parallel=True)
    def _numba_delta_encode(data):
        n = data.size
        out = np.empty(n, dtype=data.dtype)
        if n > 0:
            out[0] = data[0]
            for i in prange(1, n):
                out[i] = data[i] - data[i-1]
        return out

    @njit()
    def _numba_delta_decode(residuals):
        n = residuals.size
        out = np.empty(n, dtype=residuals.dtype)
        if n > 0:
            current = residuals[0]
            out[0] = current
            for i in range(1, n):
                current += residuals[i]
                out[i] = current
        return out
