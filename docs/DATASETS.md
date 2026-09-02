# Dataset Access

How to actually get each dataset named in `CLAUDE.md`. Access/license status was checked
2026-08-19; concrete download paths below were verified 2026-09-01 (URLs return real data as of
that date — re-verify if they start failing, hosts/paths do change).

Small spot-check samples for Phase 1 are already downloaded under `data/samples/` (gitignored —
not committed to the repo; re-run the commands below to reproduce them).

---

## Primary datasets

### Open Forest Observatory (OFO) — drone imagery, orthomosaics, CHMs, ground inventory

- **License**: CC BY 4.0. No registration.
- **Portal**: https://openforestobservatory.org/data/ (map, STAC catalog, or raw file browser)
- **Processed products (orthomosaics, DSM/mesh COGs) via STAC**: the STAC catalog only serves
  processed per-mission products, not raw frames.
  ```python
  from pystac_client import Client
  catalog = Client.open("https://stac.cyverse.org")
  search = catalog.search(collections=["Open Forest Observatory"])
  ```
  See `github.com/open-forest-observatory/stac/blob/main/scripts/STAC_API.ipynb` for the full
  filtering API (by bbox, date, platform, etc).
- **Raw per-mission images — direct HTTP, no auth**, browsable at:
  ```
  https://data.cyverse.org/dav-anon/iplant/projects/ofo/public/missions/
  ```
  470+ numbered mission folders (`000001`, `000002`, ...). Each has:
  - `images/images.zip` — the full raw overlapping-frame sequence for that mission (1.8–3.5GB
    each, checked across several missions) — a real overlapping sequence, unlike the curated
    examples below, but too large to fetch whole just for a handful of frames.
  - `images/examples/fullsize/example_{1,2,3,4}.JPG` — a handful of curated full-resolution real
    frames (~8MB each), useful for a quick detector/segmenter spot-check but **not a verified
    sequential overlapping burst** — confirmed by running SfM on them (Phase 2 prep): 0/4 frames
    registered, 0 feature matches found between any pair. Don't use these for SfM, only for
    Phase 1 wrapper sanity checks.
  - `footprint/` — mission footprint geometry.
- **Getting a real overlapping sequence without downloading the full 1.8-3.5GB zip**: the CyVerse
  host supports HTTP range requests (`Accept-Ranges: bytes`, confirmed via `curl -I`), so
  individual files can be pulled out of `images.zip` by reading its central directory remotely —
  no need to download the whole archive. The `remotezip` PyPI package does this:
  ```python
  from remotezip import RemoteZip
  import re

  url = "https://data.cyverse.org/dav-anon/iplant/projects/ofo/public/missions/000001/images/images.zip"
  with RemoteZip(url) as rz:
      names = [n for n in rz.namelist() if n.endswith(".JPG")]
      # filenames are numbered in flight-capture order, e.g. ..._000102.JPG
      frame_num = lambda n: int(re.search(r"_(\d+)\.JPG$", n).group(1))
      by_num = {frame_num(n): n for n in names}
      for fn in range(100, 113):  # a small consecutive run = real overlap
          data = rz.read(by_num[fn])
          open(f"frame_{fn:06d}.JPG", "wb").write(data)
  ```
  **Confirmed working end-to-end**: pulled mission `000001`'s frames 100-112 (13 consecutive
  frames, ~106MB total vs. the full 3.1GB archive) into
  `data/samples/ofo_mission_000001_sequence/`, ~47s over 13 range-requested files. Ran the real
  `extract_features` → `match_exhaustive` → `incremental_mapping` pipeline on them: **13/13
  frames registered, 4,653 3D points** — a genuine working reconstruction, unlike the curated
  examples above.
- **Why a second, larger sample was needed**: the 13-frame sequence above is a single narrow
  flight-line segment — confirmed via SVD of the recovered camera positions (spread ratio ~0.004
  between the smallest and largest principal axis, i.e. essentially a straight line). That's fine
  for basic SfM/scale-resolution testing, but too geometrically degenerate to robustly fit a full
  3D orientation (see `geometry/georeference.py`'s docstring and `docs/lessons_learnt.md`) — a
  real, necessary fix (`sfm.py`'s `align_gravity`) still left large horizontal-alignment residuals
  because 13 near-colinear cameras plus per-shot gravity noise just don't constrain the problem
  enough.
- **Mapping the flight pattern cheaply before downloading more frames**: rather than guessing a
  wider range, checked whether the JPEGs inside `images.zip` are stored uncompressed
  (`zipfile`/`remotezip`'s `getinfo(name).compress_type == 0`, i.e. `ZIP_STORED`) — confirmed yes.
  That means a *partial* range-read of just the first ~200KB of any single entry (via
  `RemoteZip.open(name)` as a file-like stream, `.read(200_000)`) is enough to get a real JPEG
  header with EXIF GPS, without pulling the whole ~8.6MB frame. Scanned every 5th frame across the
  full mission (77 frames, ~1s each) this way and printed lat/lon/alt for each — revealed a clean
  boustrophedon ("lawnmower") survey pattern: latitude climbs then reverses every ~25 frames
  (turns around frame 26, 51, 76, ...), with longitude stepping consistently westward between
  lines. Confirms genuine multi-line grid coverage exists in this mission, not just one strip.
  ```python
  from remotezip import RemoteZip
  from PIL import Image
  import io
  with RemoteZip(url) as rz:
      with rz.open(by_num[fn]) as f:
          data = f.read(200_000)  # header + EXIF only, not the whole ~8.6MB frame
      exif = Image.open(io.BytesIO(data))._getexif()  # GPS IFD is tag 34853
  ```
- **Sample actually used for the real (post-fix) Phase 2 geometry work**: frames 1-51 (spans the
  first flight-line turn at frame ~26, giving real cross-track/heading diversity, not just
  along-track), pulled the same `remotezip` way into
  `data/samples/ofo_mission_000001_multiline/`. This is the sample `sfm.py`,
  `scale_resolution.py`, `geometry/sfm.py`'s `align_gravity`, and `georeference.py` are validated
  against for real; the 13-frame single-line sample remains useful as the smaller/faster smoke
  test in `tests/test_scale_resolution.py`.
- **Concrete example dataset (full orthomosaic product)**:
  https://zenodo.org/records/8136161 (~10.9GB, GeoTIFF orthomosaics + GeoPackage crown
  boundaries, Danum/Sepilok Malaysia sites).
- **Ground-reference field-survey data** (species, DBH, height per tree): https://
  openforestobservatory.org/data/ground-ref/, stored as `.gpkg` point files. **Ended up not
  joined against in Phase 3, for a real reason found while building `biomass/allometry.py`**:
  this join only matters for the DBH-based Chave et al. (2014) AGB path, and DBH (trunk diameter)
  isn't observable from this project's nadir drone imagery at all — the canopy occludes the trunk
  from above. `allometry.py`'s Chave path exists (for a future ground-truthed plot with real field
  DBH), but every real tree this pipeline actually resolves goes through the crown-diameter-based
  Jucker et al. (2017) path instead, which needs species only for wood-density/functional-type
  lookup — resolved from a small USDA Forest Service table (Res. Note NRS-38, see the Phase 3
  section of `docs/lessons_learnt.md`), not this ground-reference join. Left here in case a future
  plot with real field DBH measurements makes the Chave path relevant.

**Sample downloaded** (`data/samples/ofo_mission_000001/`, ~34MB, 4 real 5472x3648 drone frames):
```bash
for i in 1 2 3 4; do
  curl -O "https://data.cyverse.org/dav-anon/iplant/projects/ofo/public/missions/000001/images/examples/fullsize/example_${i}.JPG"
done
```

### NEON — LiDAR canopy height model (gold-standard cross-check)

- **Product**: DP3.30015.001, 1m resolution GeoTIFF CHM. License: CC BY 4.0.
- **Access changed mid-2026**: now requires a free NEON account + API token (previously fully
  anonymous). Sign up at https://data.neonscience.org, generate a token, pass it in the
  `X-API-Token` header of data-portal API requests.
- **Portal**: https://data.neonscience.org/data-products/DP3.30015.001
- Also mirrored on Google Earth Engine: `projects/neon-prod-earthengine/assets/CHM_001`.
- **Not used for the actual Phase 2 validation** — checked, and NEON's AOP network has limited,
  specific site footprints; it does not cover the OFO mission `000001` site used here (~39.198°N,
  -120.904°W, Tahoe National Forest, CA). Used USGS 3DEP LiDAR instead (below), which does cover
  it. Keep this NEON entry for whichever future OFO site *does* fall inside NEON's footprint.

### USGS 3DEP LiDAR — the LiDAR ground truth actually used for Phase 2 validation

OFO mission `000001`'s site isn't inside any NEON AOP footprint (checked directly), so this
project's real LiDAR ground truth comes from USGS's 3D Elevation Program (3DEP) instead — also
freely available, also real airborne LiDAR, and it happens to have much more recent coverage for
this specific site than NEON would anyway.

- **License**: public domain (US federal government data). No registration, no API key.
- **How the exact tile(s) covering our site were found** — via USGS's TNM (The National Map)
  product-search API, which takes a bounding box and returns direct download URLs plus each
  product's own footprint (so you can confirm coverage before downloading):
  ```bash
  curl "https://tnmaccess.nationalmap.gov/api/v1/products?bbox=-120.905,39.197,-120.904,39.199&datasets=Lidar%20Point%20Cloud%20(LPC)&outputFormat=JSON"
  ```
  Returned multiple candidate LiDAR projects covering this point; picked **`CA_SierraNevada_B22`**
  (2022-2024 acquisition) over an older 2014 USFS Tahoe NF collection also available here, since
  it's much closer in time to the OFO drone imagery (2024) — less chance of real tree growth
  between the two data sources confounding the height comparison.
- **Direct LAZ tile download** (public, no auth) — the API response's `urls.LAZ` field gives a
  direct HTTPS link per 1km×1km tile, e.g.:
  ```
  https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/CA_SierraNevada_B22/CA_SierraNevada_2_2022/LAZ/USGS_LPC_CA_SierraNevada_B22_10SFJ8040.laz
  ```
  Saved to `data/lidar/` (gitignored, ~190MB for the one tile actually needed — our AOI sits
  comfortably inside a single tile, confirmed by projecting the drone frames' GPS into the tile's
  CRS before downloading, rather than guessing and downloading extra tiles speculatively).
- **A real gotcha hit here**: the first `curl` download of this file silently truncated at
  ~36.8MB (of the real 190MB) — `curl -sS` exited 0 with no visible error, and the truncated
  file's LAZ *header* parsed fine (`laspy.open()` only reads the header), which made it look
  valid until actually reading point data threw `LaszipError: reading point 0`. **Lesson: verify
  a downloaded LAZ/point-cloud file by actually reading points (`laspy.read(...)`, not just
  `laspy.open(...)`), not by header-parse success or by trusting a non-erroring `curl` exit
  code.** Re-running the same `curl` command a second time completed correctly (matched the
  server's real Content-Length) — a transient network issue, not a bad URL.
- **CRS**: confirmed directly from the downloaded file's own header (`laspy`'s
  `header.parse_crs()`), not assumed — NAD83(2011) UTM Zone 10N horizontal (EPSG:6339), NAVD88
  height via Geoid18 vertical (EPSG:5703). `geometry/georeference.py` targets EPSG:6339
  specifically because of this.
- **Classification**: standard ASPRS LAS classes — `2` = ground (used for the DTM), everything
  else (mostly class `1`, unclassified, which includes canopy returns) used for the DSM.
- **Processing**: `laspy` reads the LAZ directly in Python (`pip`-installable, no system LAZ
  library needed beyond the `laszip` extra: `laspy[laszip]`); ground points rasterized to a DTM
  via `scipy.interpolate.griddata`, non-ground points rasterized to a DSM by taking the highest
  point per grid cell (not interpolated — canopy top is a real local maximum, shouldn't be
  smoothed toward neighboring cells the way bare ground reasonably can be); CHM = DSM − DTM.
  Full logic in `scripts/spot_check_lidar_validation.py` and
  `notebooks/00_sfm_scale_validation.ipynb`.
- Considered `pdal` (the standard C++ point-cloud toolkit) for this instead of hand-rolling the
  rasterization in `laspy`/`scipy`/`numpy` — installed it via Homebrew (`brew install pdal`, ships
  as a prebuilt bottle, no compile needed) as a fallback, but ended up not needing it once the
  small-AOI `laspy` approach worked directly; left installed in case an EPT (cloud-optimized,
  streamable point cloud) query is useful later for a larger area than a single downloaded tile.

### MammAlps — camera-trap behavior video, primary activity-recognition dataset

- **License: CC BY-NC 4.0 (non-commercial)** — fine for this portfolio project, but say so
  explicitly in the top-level README since the repo is public.
- **Confirmed all-or-nothing download** — checked the Zenodo record's file listing directly via
  its API (`https://zenodo.org/api/records/15040901`): exactly 2 files, `mammalps_v1.zip`
  (≈81.8GiB) and a small `README.md`. No per-clip files, no partial-download manifest.
  ```bash
  # full dataset — only do this when actually starting Phase 5, not for a spot-check
  curl -O https://zenodo.org/records/15040901/files/mammalps_v1.zip
  ```
- DOI: 10.5281/zenodo.15040901. Code/loaders: https://github.com/eceo-epfl/MammAlps
- A newer record (https://zenodo.org/records/15588220) adds `dense_annotations_v1.zip`
  (13.5MB) — annotation metadata only, still requires the same 82GB zip for video.
- **Confirmed no partial-download workaround exists either** (checked when actually starting
  Phase 5, not just assumed): `curl -I` on the zip shows no `Accept-Ranges` header, and a real
  `Range:` request still returns `200 OK` with the full `content-length`, not `206 Partial
  Content` — Phase 2's `remotezip` trick (13 frames out of OFO's 3GB `images.zip` via HTTP range
  requests) does **not** apply to this Zenodo file server. The full 87.9GB zip is genuinely the
  only way to get more than the demo clip below.
- **Real substitute found and used for Phase 5's architecture sanity check**: the dataset's own
  GitHub repo (`github.com/eceo-epfl/MammAlps`) bundles one real, fully-annotated demo clip
  directly in git, for its own demo notebook — small enough to just download outright:
  ```bash
  mkdir -p data/samples/mammalps_demo
  BASE="https://raw.githubusercontent.com/eceo-epfl/MammAlps/main"
  curl -o data/samples/mammalps_demo/demo_video.mp4 "$BASE/resources/demo_video.mp4"              # ~12.6MB
  curl -o data/samples/mammalps_demo/demo_annotations.json "$BASE/resources/demo_annotations.json" # real dense per-frame labels
  curl -o data/samples/mammalps_demo/labels_mapping_b1.json "$BASE/evaluation/labels_mapping_b1.json"
  curl -o data/samples/mammalps_demo/labels_mapping_b2.json "$BASE/evaluation/labels_mapping_b2.json"
  ```
  One real clip (`S1_C1_E4_V0016`, 615 frames, 20.5s, 30fps), two real tracked red deer, three
  real activity classes with meaningful representation (foraging, vigilance, unknown). Used in
  place of Rolandseck (below), which turned out not to be a confirmed self-serve download either
  — see `docs/lessons_learnt.md` for the full real result from training on this clip.

---

## Camera-trap spot-check fallback (used for Phase 1, not a project dataset)

MammAlps has no small-sample path, so Phase 1's "run MegaDetector against a handful of real
frames" used a different, smaller public source instead — **this is only for sanity-checking the
wrapper code**, not a dataset this project reports results against.

### LILA BC — "Seattle(ish) Camera Traps"

- Public GCS bucket, individually downloadable clips, no account/gate.
- Informal/personal dataset (not a scientific benchmark) — fine as a wrapper smoke test, not
  something to cite as validation.
- Listing (anonymous GCS API):
  ```
  https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix=seattleish-camera-traps/converted_camera_trap_videos/
  ```

**Sample downloaded** (`data/samples/camera_trap_lila_seattleish/`, ~78MB total, 4 real
1280x720 AVI clips across 4 species/empty categories):
```bash
BASE="https://storage.googleapis.com/public-datasets-lila/seattleish-camera-traps/converted_camera_trap_videos/2018.04.15/location-06"
curl -o coyote/DSCF0004.AVI "$BASE/coyote/DSCF0004.AVI"
curl -o deer/DSCF0008.AVI   "$BASE/deer/DSCF0008.AVI"
curl -o empty/DSCF0040.AVI  "$BASE/empty/DSCF0040.AVI"
curl -o heron/DSCF0048.AVI  "$BASE/heron/DSCF0048.AVI"
```

**A `vehicle`-category clip was added later, for step 15 (alerts)'s real-footage gap.** Searched
the full bucket listing (anonymous GCS `o` list API, no delimiter, 4,464 objects total) for a
`vehicle`/`person`/`human`/`car` category folder anywhere in the dataset:
```bash
# lists every object under the whole dataset prefix, then filter names by keyword client-side
curl -s "https://storage.googleapis.com/storage/v1/b/public-datasets-lila/o?prefix=seattleish-camera-traps/converted_camera_trap_videos/&fields=items(name),nextPageToken&maxResults=1000"
```
Found **19 real `vehicle`-category clips across several dates/locations, 0 `person`/`human`
clips anywhere**. Downloaded one (`data/samples/camera_trap_lila_seattleish/vehicle/IMG_0096.AVI`,
~20MB, `2022.01.29/location-06/vehicle/`) and kept it as real evidence for a real, documented
finding — **visual inspection (this clip and two more `vehicle` clips, checked frame-by-frame,
not kept) shows no vehicle actually visible in any of them**; this dataset's `vehicle` label
appears to mark trigger events (e.g. a car on a distant road) rather than clips with a vehicle
prominently in frame. See `docs/lessons_learnt.md`'s Phase 5 section for the full finding,
including a real MegaDetector false positive found along the way. No `person`/`human` clip exists
in this dataset at all, so that branch of step 15's alert path remains real-footage-untested.

---

## Secondary / fallback datasets (not yet downloaded)

### SelvaBox — tropical tree crown segmentation

- Available now, CC BY 4.0, no gating.
- https://huggingface.co/datasets/CanopyRS/SelvaBox — 83,000+ labeled crowns, 14 orthomosaics,
  TIFF + COCO annotations. Code: `github.com/hugobaudchon/CanopyRS`, `geodataset`.
  ```bash
  huggingface-cli download CanopyRS/SelvaBox --repo-type dataset --local-dir data/selvabox
  ```

### Dronescape — secondary drone/segmentation practice set

- **Gated behind a paid IEEE DataPort subscription** ($40/mo, free only for IEEE Society
  members), for only 48MB of video (25 clips). DOI: 10.21227/t1v7-vv21.
- Given it's already scoped as secondary/weak-label in `CLAUDE.md`, **treat as skippable** unless
  the subscription is otherwise justified.

### ForestSeg — instance-level crown segmentation

- **Unconfirmed** — no self-serve data-hosting URL located (Zenodo/GitHub/Figshare all checked).
  Paper: Scientific Reports, Jan 2026, "TreeCoG" method — nature.com/articles/s41598-026-36541-y.
  Needs a manual read of the paper's Data Availability section before relying on this one.
  Already a conditional fallback in the plan, not primary — not a blocker.

### Rolandseck — 7-class deer action set (architecture sanity-check before MammAlps)

- **Unconfirmed / possibly request-based** — no public self-serve download found. One related
  paper's wording ("will be made available upon acceptance") suggests it may require emailing
  the authors. If no response, substitute another small labeled action set for the same
  sanity-check purpose (architecture validation on a tiny clean set before scaling to MammAlps).

### SA-FARI — camera-trap species-detection set

- Available. **CC BY-NC 4.0.** Annotations gated behind a Hugging Face account + click-through
  (not an approval process — just login + accept terms): `huggingface.co/facebook/SA-FARI`.
  Raw video on a public GCS bucket: `gs://cxl-public-camera-trap` (no auth needed for the video
  itself). Portal: https://www.conservationxlabs.com/sa-fari

### PanAf20K — larger wildlife species-detection set

- Available now, direct download, no application. **Non-Commercial Government Licence** —
  requires citing the dataset + its IJCV paper.
  ```bash
  curl -O https://data.bris.ac.uk/datasets/tar/1h73erszj3ckn2qjwm4sqmr2wt.zip  # 42.2GiB
  ```

---

## License summary for the public README

| Dataset | License | Commercial use |
|---|---|---|
| Open Forest Observatory | CC BY 4.0 | OK |
| NEON | CC BY 4.0 | OK |
| SelvaBox | CC BY 4.0 | OK |
| MammAlps | CC BY-NC 4.0 | **No** |
| SA-FARI | CC BY-NC 4.0 | **No** |
| PanAf20K | Non-Commercial Government Licence | **No** |
| LILA BC (Seattle(ish), spot-check only) | informal/personal, not for citation as validation | n/a |
