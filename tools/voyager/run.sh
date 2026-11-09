#!/usr/bin/env bash
# Generates the storeroom from several camera paths, then fits one splat.
#   cd /workspace/HunyuanWorld-Voyager && bash run.sh
set -euo pipefail
cd /workspace/HunyuanWorld-Voyager

PROMPT="A dim old storeroom with stained, peeling plaster walls, a stack of round metal film cans on a dark wooden trunk, a red and white railway warning sign leaning against the wall, warm low sunlight across a dusty stone floor."

(cd data_engine && python make_inputs.py --image ../room_clean.png --out ../inputs)

for dir in inputs/*/; do
  name=$(basename "$dir")
  if ls "results/$name"/*.mp4 >/dev/null 2>&1; then echo "skip $name"; continue; fi
  echo "=== $name"
  python sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "inputs/$name" \
    --prompt "$PROMPT" \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path "results/$name"
done

python to_splat.py --inputs inputs --results results --image room_clean.png --out room.ply
echo "All done: /workspace/HunyuanWorld-Voyager/room.ply"
