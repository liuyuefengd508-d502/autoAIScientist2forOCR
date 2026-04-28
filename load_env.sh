# Source this file: `source load_env.sh`
# 把 .env 中的所有 KEY=VAL 自动 export 到当前 shell

if [ ! -f .env ]; then
  echo "[load_env] .env not found in $(pwd)" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

echo "[load_env] OPENAI_API_KEY set: $([ -n \"$OPENAI_API_KEY\" ] && echo yes || echo NO)"
echo "[load_env] OPENAI_BASE_URL=$OPENAI_BASE_URL"
