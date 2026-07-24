# LaparotomySponge asset directory

Primary assets:

- `lap_sponge_unfolded.usda`: portable visual/simulation surface. Apply current surface-deformable physics at runtime.
- `lap_sponge_folded_proxy.usda`: rigid manipulation proxy with compound collisions.

Both files use relative texture references and expose a `state` variant set with `dry` and `wet` selections.

The GLB files are interchange/inspection exports. OpenUSD is the authoritative authored representation.

See `docs/LAPAROTOMY_SPONGE.md` in the DrAnmar repository for the runtime
selection, provisional parameter profile, collision audit, executed evidence,
and nonclinical provenance.

This directory is self-contained for catalog distribution and includes `LICENSE.txt` and `NOTICE.txt`.
