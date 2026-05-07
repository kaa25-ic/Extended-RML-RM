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

launcher_command=(bash "src/experiments/run_letterenv_dqn.sh" "$@")
printf -v LETTERENV_DQN_WRAPPER_COMMAND '%q ' "${launcher_command[@]}"
export LETTERENV_DQN_WRAPPER_COMMAND="${LETTERENV_DQN_WRAPPER_COMMAND% }"

tmp_dir="$(mktemp -d)"
config_src="$root/legacy/rml_reward_machines/examples/letter_env.yaml"
train_config_tmp="$tmp_dir/letter_env_train.yaml"
eval_config_tmp="$tmp_dir/letter_env_eval.yaml"
train_monitor_log="$tmp_dir/train_monitor.log"
eval_monitor_log="$tmp_dir/eval_monitor.log"
train_monitor_pid=""
eval_monitor_pid=""
train_monitor_port="${MONITOR_PORT:-18081}"
eval_monitor_port="${EVAL_MONITOR_PORT:-18082}"
monitor_script="${MONITOR_SCRIPT_NAME:-online_monitor_edit.sh}"
monitor_spec="${MONITOR_SPEC_NAME:-./letter_env_spec_numerical_runtime_compatible.pl}"

cleanup() {
  if [[ -n "$train_monitor_pid" ]]; then
    pkill -P "$train_monitor_pid" >/dev/null 2>&1 || true
    kill "$train_monitor_pid" >/dev/null 2>&1 || true
    wait "$train_monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "$eval_monitor_pid" ]]; then
    pkill -P "$eval_monitor_pid" >/dev/null 2>&1 || true
    kill "$eval_monitor_pid" >/dev/null 2>&1 || true
    wait "$eval_monitor_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp_dir"
}

trap cleanup EXIT

if [[ "$train_monitor_port" == "$eval_monitor_port" ]]; then
  echo "Training and evaluation monitor ports must be different." >&2
  exit 1
fi

sed "s/^port: .*/port: ${train_monitor_port}/" "$config_src" > "$train_config_tmp"
sed "s/^port: .*/port: ${eval_monitor_port}/" "$config_src" > "$eval_config_tmp"

(
  cd "$root/legacy/RML"
  tail -f /dev/null | "./$monitor_script" "$monitor_spec" "$train_monitor_port" >"$train_monitor_log" 2>&1
) &
train_monitor_pid=$!
(
  cd "$root/legacy/RML"
  tail -f /dev/null | "./$monitor_script" "$monitor_spec" "$eval_monitor_port" >"$eval_monitor_log" 2>&1
) &
eval_monitor_pid=$!

for _ in $(seq 1 20); do
  if bash -c "exec 3<>/dev/tcp/127.0.0.1/${train_monitor_port}" >/dev/null 2>&1 \
    && bash -c "exec 3<>/dev/tcp/127.0.0.1/${eval_monitor_port}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! bash -c "exec 3<>/dev/tcp/127.0.0.1/${train_monitor_port}" >/dev/null 2>&1; then
  echo "Training monitor failed to start. Log follows:" >&2
  cat "$train_monitor_log" >&2
  exit 1
fi

if ! bash -c "exec 3<>/dev/tcp/127.0.0.1/${eval_monitor_port}" >/dev/null 2>&1; then
  echo "Evaluation monitor failed to start. Log follows:" >&2
  cat "$eval_monitor_log" >&2
  exit 1
fi

cd "$root"
"$python_bin" -m src.experiments.train_letterenv_dqn \
  --config "$train_config_tmp" \
  --eval-config "$eval_config_tmp" \
  "$@"
