"""Safe failure codes for background jobs.

Stored as plain strings on the job's row and shown to the client, so they must
never carry provider text. Add a code here before writing it anywhere.
"""

import enum


class JobErrorCode(enum.StrEnum):
    # The PDF parsed but produced no text (image-only scans); a retry cannot help.
    NO_TEXT = "no_text"
    # The stored upload is gone from disk; a retry cannot help.
    FILE_MISSING = "file_missing"
    # The model or embedding provider failed; a retry usually helps.
    PROVIDER_ERROR = "provider_error"
    # The worker holding the job died or was redeployed and its lease ran out.
    WORKER_LOST = "worker_lost"
    # Automatic re-runs are used up; only a person can spend more.
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    # The job never reached a worker (Redis was down or lost the message).
    QUEUE_UNAVAILABLE = "queue_unavailable"


# Failures where another attempt has a real chance of succeeding.
RETRYABLE_ERROR_CODES = frozenset(
    {
        JobErrorCode.PROVIDER_ERROR,
        JobErrorCode.WORKER_LOST,
        JobErrorCode.ATTEMPTS_EXHAUSTED,
        JobErrorCode.QUEUE_UNAVAILABLE,
    }
)
