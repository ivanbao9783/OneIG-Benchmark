#!/bin/bash

# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN

# image_root_dir
IMAGE_DIR=""

# model list
MODEL_NAMES=("Qwen-Image")
# MODEL_NAMES=("gpt-4o" "imagen4")

# image grid
IMAGE_GRIDS=("2,2")
# IMAGE_GRIDS=("2,2" "1,4")

pip install transformers==4.57.0

# Style Score

echo "It's style time."

python -m scripts.style.style_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR/anime" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRIDS[@]}" \

rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."