"""Verified access to the vendored official ReinFlow source tree."""

from .vendor_identity import (
    EXPECTED_REINFLOW_COMMIT,
    EXPECTED_VENDOR_FILE_COUNT,
    EXPECTED_VENDOR_MANIFEST_SHA256,
    VendorIdentity,
    verify_vendor_snapshot,
)
from .symbols import (
    OfficialImportError,
    OfficialReinFlowSymbols,
    load_official_symbols,
)

__all__ = [
    "EXPECTED_REINFLOW_COMMIT",
    "EXPECTED_VENDOR_FILE_COUNT",
    "EXPECTED_VENDOR_MANIFEST_SHA256",
    "VendorIdentity",
    "verify_vendor_snapshot",
    "OfficialImportError",
    "OfficialReinFlowSymbols",
    "load_official_symbols",
]
