# Dual Scorpion source assets

This directory contains the unmodified Genesis model subtree from
[`momoiorg-repository/dual_scorpion`](https://github.com/momoiorg-repository/dual_scorpion),
subtree `dual_scorpion_Full_urdf/genesis`, pinned at upstream `main` commit
`1f64c14bfe67f957b3c718ca9f9b06965502bd51`.

The URDF and all 392 companion files were compared with that exact upstream revision.
The 17 STL files stored by upstream through Git LFS were verified by SHA-256 and size
against the LFS pointers; the other files were byte-identical. The URDF SHA-256 is
`94097ff255ded808ccb287344d9d4fc28618388debcb2808e7a1bbd904b02a48`.
`SHA256SUMS` pins every file in the included Genesis subtree. `SOURCE.json` records
provenance and the upstream revision. The upstream Apache-2.0 license is included as
`LICENSE`.

The subtree is model data, not a claim of hardware calibration or physical task success.
Keep upstream source files unchanged; prepare or import derived artifacts into the
project's ignored generated-robot directory.
