# Legacy alignment

The former ASR/CTC-first alignment path is retained in `backend/processing/`
as a compatibility fallback. Production now selects the SOFA lane by default
(`ALIGNMENT_PRIMARY_BACKEND=sofa`); set `ALIGNMENT_PRIMARY_BACKEND=legacy` or
`ALIGNMENT_SOFA_BACKEND=off` only when an installation cannot use the local
SOFA checkpoint.

This directory is archival and is not imported by the application. The source
files remain in their original locations so persisted jobs and older worker
payloads can still be read without a destructive migration.
