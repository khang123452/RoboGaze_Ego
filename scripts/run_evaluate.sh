export GEMINI_API_KEY=

uv run python scripts/evaluate.py \
  --pred-dir outputs/robogaze_dataset \
  --gt-dir ground_truth \
  --out-dir report/robogaze_eval \
  --model gemini-3.1-flash-lite