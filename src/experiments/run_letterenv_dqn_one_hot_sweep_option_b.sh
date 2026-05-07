#!/usr/bin/env bash
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

monitor_script="${MONITOR_SCRIPT_NAME:-online_monitor_edit_fast.sh}"
base_output_dir="${SWEEP_OUTPUT_DIR:-$root/src/results/dqn_letterenv/one_hot/sweep_option_b_n1_seed0_20k}"

seed=0
n_value=1
total_timesteps=20000
eval_freq=5000
n_eval_episodes=5

exploration_fractions=(0.3 0.5)
learning_starts_values=(500 1000)
learning_rates=(0.0005 0.001 0.002)
batch_sizes=(32 64)

mkdir -p "$base_output_dir"

manifest_path="$base_output_dir/sweep_manifest.csv"
cat >"$manifest_path" <<'CSV'
run_name,seed,n_value,total_timesteps,exploration_fraction,learning_starts,learning_rate,batch_size,eval_freq,n_eval_episodes,monitor_script
CSV

sanitize_float() {
  local value="$1"
  echo "${value//./p}"
}

run_index=0
total_runs=$(( ${#exploration_fractions[@]} * ${#learning_starts_values[@]} * ${#learning_rates[@]} * ${#batch_sizes[@]} ))

for exploration_fraction in "${exploration_fractions[@]}"; do
  for learning_starts in "${learning_starts_values[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        run_index=$((run_index + 1))

        exp_tag="$(sanitize_float "$exploration_fraction")"
        lr_tag="$(sanitize_float "$learning_rate")"
        run_name="seed${seed}_n${n_value}_t${total_timesteps}_exp${exp_tag}_ls${learning_starts}_lr${lr_tag}_bs${batch_size}"
        output_dir="$base_output_dir/$run_name"

        printf '%s\n' "$run_name,$seed,$n_value,$total_timesteps,$exploration_fraction,$learning_starts,$learning_rate,$batch_size,$eval_freq,$n_eval_episodes,$monitor_script" >>"$manifest_path"

        if [[ -f "$output_dir/summary.json" ]]; then
          echo "[$run_index/$total_runs] Skipping completed run: $run_name"
          continue
        fi

        echo "[$run_index/$total_runs] Starting run: $run_name"
        env MONITOR_SCRIPT_NAME="$monitor_script" PYTHON_BIN="$python_bin" \
          bash "$here/run_letterenv_dqn.sh" \
          --encoding one_hot \
          --n-value "$n_value" \
          --seed "$seed" \
          --total-timesteps "$total_timesteps" \
          --learning-starts "$learning_starts" \
          --learning-rate "$learning_rate" \
          --batch-size "$batch_size" \
          --exploration-fraction "$exploration_fraction" \
          --eval-freq "$eval_freq" \
          --n-eval-episodes "$n_eval_episodes" \
          --output-dir "$output_dir"
      done
    done
  done
done

echo "Completed sweep. Results root: $base_output_dir"
