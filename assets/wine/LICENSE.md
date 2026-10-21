# Wine vessels — CC0 1.0 Universal

Author: Innerscene, original unbranded models. Downloaded 2026-10-20.

- Wine bottle: https://www.innerscene.com/tools/library/3d-parts/wine-bottle-original-drink-vessel-dff39232
- Wine glass: https://www.innerscene.com/tools/library/3d-parts/wine-glass-original-drink-vessel-98bf1e17
- Library terms: https://www.innerscene.com/tools/library/terms
- License: https://creativecommons.org/publicdomain/zero/1.0/

Both source pages identify these original models as CC0 and expressly allow use,
modification and redistribution, including commercially, without attribution.
The original GLBs are included unchanged, with SHA-256 hashes in provenance.json.
The project keeps this attribution and provenance voluntarily.

Derived OBJ visuals rotate Y-up to Z-up, scale the vessels to the OpenArm workspace,
and remove the bottle's cork. Transparent visuals use a single outward-facing skin
to avoid per-triangle blending artifacts in the native rasterizer. Collision meshes reconstruct the measured rotational
inner/outer profiles as watertight material shells, retaining the open cavities.
scripts/prepare_wine_assets.py reproduces these derivatives from the original files.
These changes are simulation geometry preparation, not a new robot model.

CC0 dedicates copyright and related rights to the public domain to the extent
permitted by law, with the fallback license in the linked CC0 legal text. Assets are
provided without warranty. No trademark or patent rights are granted by CC0.
