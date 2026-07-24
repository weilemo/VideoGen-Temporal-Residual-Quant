# Reference Sources

This directory contains read-only upstream snapshots used for provenance and
comparison. New TRQ behavior must be implemented in `temporalresidualkvquant/`
or `integrations/`, not directly in a reference tree.

- `quant-videogen/`: original Quant-VideoGen snapshot, including the
  HY-WorldPlay source currently used by its integration.
- `focused-forcing-code/`: focused-forcing variants retained for architecture
  and head-importance provenance.

The existing HY-WorldPlay hook inside `quant-videogen/` is a compatibility
exception. Future upstream modifications should be represented as a patch in
the matching integration directory.
