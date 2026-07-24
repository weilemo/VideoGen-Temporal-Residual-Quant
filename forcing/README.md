# Legacy Forcing Runtimes

These directories are upstream/runtime snapshots retained for compatibility
with the remote GPU checkout. They are not the ownership boundary for TRQ.
New quantization code belongs in `temporalresidualkvquant/src/trq/`; launchers
and source patches belong in `integrations/`.

| Directory | Actual role |
| --- | --- |
| `selfforcing/` | legacy Self-Forcing runtime; active vendored backend is under `temporalresidualkvquant/backends/` |
| `rollingforcing/` | Rolling Forcing runtime currently used by its integration |
| `causalforcing/` | legacy minute-level Rolling/Long-video variant, despite its historical name |
| `longlive/` | LongLive upstream runtime |
| `vbench/` | VBench runtime |
| `longcatvideo/` | ignored remote/local upstream clone created by the LongCat setup script |

Official Causal Forcing is a separate remote clone installed and patched by
`integrations/causal_forcing/setup.sh`. Do not treat `forcing/causalforcing/`
as the official Causal Forcing backend.
