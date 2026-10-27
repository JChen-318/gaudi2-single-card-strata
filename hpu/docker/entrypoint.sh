#!/usr/bin/env bash
# Container start: fill the engine config from the environment, start the server, warm it up once it is ready.
#   MODEL_GGUF    first shard of the IQ4_XS GGUF            (default /models/qwen3.8-flash-next-uncensored-iq4xs/...)
#   DFLASH_DIR    DFlash2 draft model directory             (default /models/qwen3.8-flash-next-dflash2)
#   TOKENIZER_DIR tokenizer directory                       (default /tokenizer)
#   STRATA_CONFIG config template                           (default hpu/docker/strata-hpu.json)
#   STRATA_PORT   HTTP port (8080); API_KEY: required key for clients (none by default)
#   WARMUP        1 (default): after READY, compile the prompt-tail window graphs with three prompts
set -euo pipefail
S=/opt/strata
TEMPLATE=${STRATA_CONFIG:-$S/hpu/docker/strata-hpu.json}
mkdir -p /var/log/strata
python3 - "$TEMPLATE" /tmp/strata-hpu.json <<'PY'
import json,os,sys
defaults={"MODEL_GGUF":"/models/qwen3.8-flash-next-uncensored-iq4xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf",
          "DFLASH_DIR":"/models/qwen3.8-flash-next-dflash2","TOKENIZER_DIR":"/tokenizer"}
text=open(sys.argv[1]).read()
for k,v in defaults.items(): text=text.replace("${%s}"%k,os.environ.get(k,v))
c=json.loads(text)
if os.environ.get("API_KEY"): c["api_key"]=os.environ["API_KEY"]
for k in ("MODEL_GGUF","DFLASH_DIR","TOKENIZER_DIR"):
    p=os.environ.get(k,defaults[k])
    if not os.path.exists(p): sys.exit("strata-gaudi2: %s=%s does not exist in the container (mount it)"%(k,p))
json.dump(c,open(sys.argv[2],"w"),indent=1)
PY
if [ "${WARMUP:-1}" = 1 ]; then
  python3 "$S/hpu/docker/warmup.py" "http://127.0.0.1:${STRATA_PORT}" &
fi
cd "$S"
exec python3 -m serve.server --engine strata --config /tmp/strata-hpu.json --host 0.0.0.0 --port "${STRATA_PORT}"
