# E-D3DGS-RGBT

This repository contains the reproducible implementation of dynamic RGB--Thermal Gaussian reconstruction with blind temporal alignment and effective cross-modal spatial correction. The spatial variable is an effective rig-level correction under the shared virtual camera model used by the data loader; it is not a claim of metrically accurate physical Thermal intrinsics or extrinsics.

## Supported data

The loader expects the Nerfies-style dual-modality layout below. `scene.json`, `metadata.json`, and `dataset.json` are required; RGB and Thermal frames are paired by the IDs in `dataset.json` and camera metadata are read from `camera/<id>.json`.

```
<dataset>/
  scene.json
  metadata.json
  dataset.json
  camera/<id>.json
  images/1x/<id>.png
```

Official scene configurations are:

| Family | Scene names | Configuration |
|---|---|---|
| Lab | `Heatingtable`, `HotPressMachine`, `Hotwind` | `arguments/<scene>.py` |
| MeetingRoom | `Bacon`, `covers`, `DeliverIcePacks`, `HairDryer`, `HairDryerDark`, `IroningClothes`, `LightTheCandles`, `PourHotWater`, `WhiteFoamCovers` | `arguments/<scene>.py` |

Each of these 12 scene names has a matching RGB-teacher entrypoint at
`arguments/<scene>_rgb_teacher.py`. `covers.py` and `PourHotWater.py` keep their
dedicated implementations; the other per-scene files are thin, validated
wrappers around the shared MeetingRoom configuration, so frame counts and
strict training rules cannot drift between scenes.

This is support for the listed scene configurations and their required data schema, not arbitrary datasets. A new scene needs a matching configuration and a fixed-support contract generated with `time_alignment/build_fixed_support.py`. For formal shift experiments, a scene-specific contract is required; the launcher fails closed when it is missing.

## Environment

Use Python 3.10 with a CUDA-enabled PyTorch installation compatible with the host driver. The verified development environment used PyTorch 1.13.1 with CUDA 11.6. Install the Python dependencies from `requirements.txt`, then build both local CUDA extensions:

```bash
pip install -r requirements.txt
pip install -e submodules/3dgs-pose
pip install -e submodules/simple-knn
```

The exact CUDA toolkit path can be supplied through `ED3DGS_CUDA_HOME`. The launcher does not assume a host name, Conda path, or local absolute symlink.

## Running

All actions use the same entry point. Set the dataset explicitly and keep each output root isolated.

```bash
export ED3DGS_SCENE=covers
export ED3DGS_DATASET=/path/to/covers
export ED3DGS_EXPERIMENT_SHIFT=20
export ED3DGS_TEACHER_MODEL_PATH=/path/to/artifacts/covers/rgb_teacher

bash run.sh teacher 0 6666
bash run.sh lifecycle 0 6666
bash run.sh verify
```

For Lab and MeetingRoom scenes, the launcher automatically selects the matching per-scene full and RGB-teacher configurations. Set `ED3DGS_SUPPORT_CONTRACT` when using a generated or externally stored contract. The default formal protocol is strict V2: a 30,000-step reconstruction block plus a 15,000-step calibration block (45,000 outer iterations), with scene variables frozen during calibration. Gate actions use the selected smaller scene-step budget and the same ownership rules. `run.sh` refuses existing output roots and never overwrites a teacher, Gate, or evaluation artifact.

The RGB teacher is trained from scratch for 30,000 iterations with Thermal losses, pose, intrinsics, temporal alignment, and modality updates disabled. Stage 2 starts from a raw-zero clock and consumes only the saved teacher artifact. Formal evaluation is fixed-support, pure forward, and test-time optimization is disabled.

## Repository layout

`train.py`, `render.py`, and `metrics.py` are the public Python entry points. `arguments/` contains scene and support contracts; `scene/`, `gaussian_renderer/`, `utils/`, and `time_alignment/` contain runtime code and audits; `submodules/` contains the local CUDA extension sources. Generated checkpoints, renders, logs, and teachers are external artifacts and are intentionally excluded from this source release.

## Reproducibility and scope

Record the scene, dataset path, support-contract path and SHA-256 manifest for every run. Results are protocol-relative: compare runs with the same scene, split, seed, rasterizer build, support contract, optimizer-step budget, and evaluation script. Thermal intrinsics remain fixed in the released method. Any claim about physical camera calibration requires an independent camera-truth experiment; the released method reports effective RGB-to-Thermal alignment under a shared virtual camera model.

## License

The upstream Gaussian-Splatting research license in `LICENSE.md` applies to the corresponding code. Review the licenses of the CUDA extensions and dependencies before redistribution.
