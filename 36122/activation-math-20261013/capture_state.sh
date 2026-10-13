#!/bin/bash
set -euo pipefail
probe_dir=/home/liuhao/sglang-pr-assets/36122/math-validation-20261013.mmzczj
{
  date --iso-8601=seconds
  hostname
  git -C /home/liuhao/sglang-worktrees/issue-35019 status --short --branch
  git -C /home/liuhao/sglang-worktrees/issue-35019 rev-parse HEAD
  git -C "$probe_dir/main" rev-parse HEAD
  git -C "$probe_dir/integrated" show -s --format='%H %P %T' HEAD
  git -C "$probe_dir/integrated" diff --shortstat 7ff3a112846a3fcc12a27bba09d644885a226e4d HEAD
  git -C "$probe_dir/integrated" diff --check 7ff3a112846a3fcc12a27bba09d644885a226e4d HEAD
  docker image inspect quay.io/gpustack/runner:cuda13.0-sglang0.5.15.post1 --format '{{.Id}} {{json .RepoDigests}}'
  nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,utilization.memory --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  free -h
  docker ps --format '{{.Names}} {{.Status}}'
  curl --noproxy '*' --silent --show-error --max-time 5 -o /dev/null -w 'existing_vllm_health_http=%{http_code}\n' http://127.0.0.1:8101/health
} > "$probe_dir/state-final.log"
