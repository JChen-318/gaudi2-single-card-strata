# Warm a freshly started server: wait for the model, then send prompts of exactly 8191, 4095 and 2047 tokens, whose
# prefill tails cover every power-of-two window size, so later requests do not compile those graphs on first use.
#   python3 warmup.py http://127.0.0.1:8080
import json,sys,time,urllib.request
base=sys.argv[1] if len(sys.argv)>1 else "http://127.0.0.1:8080"
def get(path):
    return json.load(urllib.request.urlopen(base+path,timeout=10))
model=None
for _ in range(720):
    try:
        data=get("/v1/models")["data"]
        if data and data[0].get("status",{}).get("value","loaded")=="loaded": model=data[0]["id"];break
    except Exception: pass
    time.sleep(5)
if not model: sys.exit("warmup: the server did not become ready")
def send(text):
    body=json.dumps({"model":model,"temperature":0,"max_tokens":8,"messages":[{"role":"user","content":text}]}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
    return r["usage"]["prompt_tokens"]
filler=lambda k:" ".join("The quick brown fox jumps over the lazy dog number %d."%i for i in range(k))
start=time.time()
for target in (8191,4095,2047):
    k=int((target-40)/14.86);extra=0
    for _ in range(6):
        tokens=send(filler(k)+" x"*extra)
        if tokens==target: break
        extra+=target-tokens
        if extra<0: k-=1;extra+=15
print("strata-gaudi2: warm-up done in %.0f s"%(time.time()-start),flush=True)
