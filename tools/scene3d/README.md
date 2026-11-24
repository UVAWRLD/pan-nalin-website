# The 3D hero scene

The hero photo (`assets/img/hero-boy-film-strip.png`) is turned into a scene
the camera can move through a little: the photo cut into depth layers, each a
picture with transparency plus a depth map, drawn by `scene3d.js` as relief
meshes seen from the photo's own camera.

| layer      | what it holds                                                        |
|------------|----------------------------------------------------------------------|
| `surround` | the room continued past the photo's edges (painted, half resolution) |
| `far`      | wall and floor; under everything else, painted in                    |
| `objects`  | the trunk and the film cans                                          |
| `boy`      | the boy and his film; behind his near arm, a painting of what it hid |
| `front`    | his near hand and forearm, and the strip hanging in front of him     |

From the photo's viewpoint the layers add up to the photo exactly (the build
prints the error; it should stay well under one level out of 255). From
anywhere else, whatever moves aside uncovers painted-in picture, not holes.

## Rebuilding

Python 3.12 with the packages in `requirements.txt` (PyTorch with Apple's MPS
works; models download from Hugging Face on first use).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r tools/scene3d/requirements.txt
.venv/bin/python tools/scene3d/prepare.py   # depth, cut-outs (~10 min, once)
.venv/bin/python tools/scene3d/build.py     # layers into assets/scene/ (~2 min; first run ~25 min)
```

`prepare.py` and the slow painting steps in `build.py` keep their results in
`work/` (git-ignored) and reuse them. Delete a file in `work/cache/` to redo
that step. Painting steps use random seeds and are picked by eye:
`HIDDEN_SEED` in `build.py` chose the painting behind the near arm.

Inputs besides the photo: `tools/voyager/room_clean.png` (the room with the
boy painted out) and `prompts_room.json` (points that pick out the trunk and
cans for the segmentation model).

## Checking

`shots.mjs` renders `scene.html` from any camera positions in headless Chrome
(Playwright, installed under `work/qa/`); `hero_shots.mjs` does the same for
the home page; `sheet.py` puts renders side by side.

```bash
node tools/scene3d/shots.mjs work/shots/test 1920 1200 views.json
```

`scene.html` shows the scene on its own. Its query options: `?still&yaw=-6&pitch=2&dolly=0.5`
holds a view, `?only=front,boy` draws only those layers, `?half` uses the phone pictures.

## Tools

- `depth.py`: metric depth and focal length (Apple Depth Pro)
- `segment.py`: masks from point prompts (Segment Anything)
- `matte.py`: alpha mattes from trimaps, tiled for 4K (ViTMatte)
- `outpaint.py`: the room beyond the frame and the area behind the arm (Stable Diffusion inpainting)
- `common.py`: fills and the plain inpainting model (big-LaMa)
