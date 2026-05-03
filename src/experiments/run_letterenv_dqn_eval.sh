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

tmp_dir="$(mktemp -d)"
config_src="$root/legacy/rml_reward_machines/examples/letter_env.yaml"
config_tmp="$tmp_dir/letter_env.yaml"
monitor_log="$tmp_dir/monitor.log"
monitor_pid=""
monitor_port="${MONITOR_PORT:-18081}"
monitor_script="${MONITOR_SCRIPT_NAME:-online_monitor_edit.sh}"
monitor_spec="${MONITOR_SPEC_NAME:-./letter_env_spec_numerical_runtime_compatible.pl}"

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

for _ in $(seq 1 20); do
  if bash -c "exec 3<>/dev/tcp/127.0.0.1/${monitor_port}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! bash -c "exec 3<>/dev/tcp/127.0.0.1/${monitor_port}" >/dev/null 2>&1; then
  echo "Monitor failed to start. Log follows:" >&2
  cat "$monitor_log" >&2
  exit 1
fi

cd "$root"
"$python_bin" -m src.experiments.evaluate_letterenv_dqn_policy --config "$config_tmp" "$@"
