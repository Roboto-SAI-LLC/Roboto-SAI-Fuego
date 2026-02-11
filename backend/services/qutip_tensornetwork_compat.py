"""
Compatibility shim to provide a lightweight MatrixProductState and mps_apply
for environments where qutip.tensornetwork is not available. This is a
minimal functional shim that uses qutip dense Qobj operations and does not
provide actual MPS performance benefits — it's intended for testing and CI.
"""
from __future__ import annotations
from typing import List
try:
    import qutip
    from qutip import Qobj
except Exception:
    qutip = None
    Qobj = None


class MatrixProductState:
    """Lightweight wrapper that mimics API used by the repo.

    Only implements `from_ket` and stores the dense state vector as numpy.
    """
    def __init__(self, ket: Qobj):
        self._ket = ket

    @staticmethod
    def from_ket(ket: Qobj):
        return MatrixProductState(ket)

    def to_ket(self) -> Qobj:
        return self._ket

    def full(self):
        return self._ket.full()


def mps_apply(ops: List[Qobj], mps: MatrixProductState) -> MatrixProductState:
    """Apply dense qutip operators sequentially to the underlying ket.

    This is a dense fallback and does not preserve MPS efficiency.
    """
    if qutip is None:
        raise RuntimeError("qutip not available for fallback mps_apply")

    ket = mps.to_ket()
    # Apply each operator in the provided list (operators are qutip.Qobj)
    result = ket
    for op in ops:
        result = op * result

    return MatrixProductState(result)


__all__ = ["MatrixProductState", "mps_apply"]
