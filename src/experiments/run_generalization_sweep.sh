#!/usr/bin/env bash
# Zero-shot generalization sweep: evaluate trained models on unseen N values.
#
# Usage (from repo root):
#   bash src/experiments/run_generalization_sweep.sh \
#     --results-root src/results/dqn_letterenv/one_hot \
#     --run-pattern "n1_progress_shaping_expfrac04_eval20_500k_seed*" \
#     --eval-n-values "1 2 3 4 5" \
#     --episodes 20 \
#     --output-dir src/results/generalization/one_hot_n1_transfer
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root="$(cd "$here/../.." >/dev/null 2>&1 && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  python_bin="$(command -v python)"
fi

# ---------- defaults ----------
results_root="src/results/dqn_letterenv/one_hot"
run_pattern="n1_progress_shaping_expfrac04_eval20_500k_seed*"
eval_n_values="1 2 3 4 5"
episodes=20
output_dir="src/results/generalization/one_hot_n1_transfer"
model_kind="best"
monitor_port="${MONITOR_PORT:-18081}"
monitor_script="${MONITOR_SCRIPT_NAME:-online_monitor_edit.sh}"
monitor_spec="${MONITOR_SPEC_NAME:-./letter_env_spec_numerical_runtime_compatible.pl}"

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root)    results_root="$2";    shift 2 ;;
    --run-pattern)     run_pattern="$2";     shift 2 ;;
    --eval-n-values)   eval_n_values="$2";   shift 2 ;;
    --episodes)        episodes="$2";        shift 2 ;;
    --output-dir)      output_dir="$2";      shift 2 ;;
    --model-kind)      model_kind="$2";      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------- monitor setup ----------
tmp_dir="$(mktemp -d)"
config_src="$root/legacy/rml_reward_machines/examples/letter_env.yaml"
config_tmp="$tmp_dir/letter_env.yaml"
monitor_log="$tmp_dir/monitor.log"
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    pkill -P "$monitor_pid" >/dev/null 2>&1 || true
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

sed "s/^port: .*/port: ${monitor_port}/" "$config_src" > "$config_tmp"

(
  cd "$root/legacy/RML"
  tail -f /dev/null | "./$monitor_script" "$monitor_spec" "$monitor_port" >"$monitor_log" 2>&1
) &
monitor_pid=$!

echo "Waiting for monitor on port ${monitor_port}..."
for _ in $(seq 1 20); do
  if bash -c "exec 3<>/dev/tcp/127.0.0.1/${monitor_port}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! bash -c "exec 3<>/dev/tcp/127.0.0.1/${monitor_port}" >/dev/null 2>&1; then
  echo "Monitor failed to start. Log:" >&2
  cat "$monitor_log" >&2
  exit 1
fi
echo "Monitor ready."

# ---------- sweep ----------
cd "$root"
mkdir -p "$output_dir"

for run_dir in $results_root/$run_pattern; do
  [[ -d "$run_dir" ]] || continue
  run_name="$(basename "$run_dir")"

  for n in $eval_n_values; do
    out="$output_dir/${run_name}_n${n}"
    echo "--- Evaluating $run_name on N=$n ---"
    "$python_bin" -m src.experiments.evaluate_letterenv_dqn_policy \
      --run-dir "$run_dir" \
      --config "$config_tmp" \
      --model-kind "$model_kind" \
      --n-value "$n" \
      --episodes "$episodes" \
      --max-steps 200 \
      --seed-base 0 \
      --reseed-each-episode \
      --output-dir "$out"
  done
done

# ---------- aggregate ----------
echo ""
echo "=== Generalization Results ==="
printf "%-55s %6s %12s %12s\n" "run" "N" "success_rate" "mean_return"
for summary_file in "$output_dir"/*/summary.json; do
  [[ -f "$summary_file" ]] || continue
  run=$(basename "$(dirname "$summary_file")")
  n=$("$python_bin" -c "import json,sys; d=json.load(open('$summary_file')); print(d['eval_n_value'])")
  sr=$("$python_bin" -c "import json,sys; d=json.load(open('$summary_file')); print(f\"{d['success_rate']:.2f}\")")
  mr=$("$python_bin" -c "import json,sys; d=json.load(open('$summary_file')); print(f\"{d['mean_total_reward']:.1f}\")")
  printf "%-55s %6s %12s %12s\n" "$run" "$n" "$sr" "$mr"
done
