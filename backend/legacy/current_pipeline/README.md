# Pre-SOFA pipeline archive

This folder marks the former ASR/CTC-first production approach as legacy. Its
implementation is intentionally still present under `backend/processing/` as a
non-destructive compatibility fallback for old jobs and worker payloads; the
selector no longer chooses it by default. Use `ALIGNMENT_PRIMARY_BACKEND=legacy`
to opt back in for a diagnostic comparison.
