#!/bin/bash
# 채점 모델 1개를 로컬 OpenAI 호환 서버로 올린다(127.0.0.1:8000, 한 번에 1개). 사용: serve.sh gpt-oss|qwen3|exaone|stop
set -e
V=~/chat-eval/venv_judge/bin
S=kcs-mjng-eval-v8-serve
case "$1" in
  gpt-oss) M=openai/gpt-oss-120b; NAME=gpt-oss-120b; NEED=85; ARGS="--gpu-memory-utilization 0.70" ;;
  qwen3)   M=Qwen/Qwen3-32B-FP8; NAME=qwen3-32b-fp8; NEED=55; ARGS="--gpu-memory-utilization 0.45" ;;
  exaone)  M=LGAI-EXAONE/EXAONE-4.0-32B; NAME=exaone-4.0-32b; NEED=85; ARGS="--gpu-memory-utilization 0.70" ;;
  stop) tmux kill-session -t $S 2>/dev/null; sleep 5; pgrep -f 'vllm serve' || echo stopped; exit 0 ;;
  *) echo "usage: $0 gpt-oss|qwen3|exaone|stop"; exit 1 ;;
esac
tmux has-session -t $S 2>/dev/null && { echo "이미 서버가 떠 있음 - 먼저 stop"; exit 1; }
AVAIL=$(free -g | awk '/^Mem|^메모리/{print $7}')
[ "$AVAIL" -lt "$NEED" ] && { echo "가용 메모리 ${AVAIL}G < 필요 ${NEED}G - 다른 작업과 겹침, 시작 안 함"; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
REV=$(cat ~/.cache/huggingface/hub/models--${M//\//--}/refs/main)
tmux new -d -s $S "PATH=$V:/usr/local/cuda/bin:\$PATH CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1 $V/vllm serve $M --revision $REV --served-model-name $NAME --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --max-num-seqs 8 --seed 42 $ARGS > ~/chat-eval/judge/serve_$NAME.log 2>&1"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && { echo "READY $NAME rev=$REV after $((i*10))s"; free -g | head -2; exit 0; }
  tmux has-session -t $S 2>/dev/null || { echo "서버 종료됨"; grep -iE 'error|exception' ~/chat-eval/judge/serve_$NAME.log | tail -5 | cut -c1-200; exit 1; }
  sleep 10
done
echo "20분 안에 안 뜸"; exit 1
