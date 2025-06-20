from enum import Enum

class VerificationStatus(Enum):
    """Verification status enumeration"""
    PENDING = "pending"
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"
    PARTIALLY_TRUE = "partially_true"
    DISPUTED = "disputed"
    UNVERIFIABLE = "unverifiable"
    ERROR = "error"
