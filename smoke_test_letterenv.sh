#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$ROOT/rml_reward_machines/.venv/bin/python" ]]; then
  python_bin="$ROOT/rml_reward_machines/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  python_bin="$(command -v python)"
fi

tmp_dir="$(mktemp -d)"
config_src="$ROOT/rml_reward_machines/examples/letter_env.yaml"
config_tmp="$tmp_dir/letter_env_smoke.yaml"
monitor_log="$tmp_dir/monitor.log"
monitor_pid=""
monitor_port=18081

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
  cd "$ROOT/RML"
  tail -f /dev/null | ./online_monitor_edit.sh ./letter_env_spec_numerical.pl "$monitor_port" >"$monitor_log" 2>&1
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

cd "$ROOT/rml_reward_machines"
"$python_bin" ./scripts/smoke_test_letterenv.py --config "$config_tmp"

echo "Smoke test passed."
