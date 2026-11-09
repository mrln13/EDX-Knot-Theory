"""Extract the first run (file name + cumulative composition series) from a large aggregate JSON
without loading the whole file. usage: python first_run.py IN.json OUT.json [MB]"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
mb = int(sys.argv[3]) if len(sys.argv) > 3 else 12
with open(src, "r", encoding="utf-8") as f:
    buf = f.read(mb * 1024 * 1024)
i = buf.index('{"file"')
run, end = json.JSONDecoder().raw_decode(buf, i)
print(run["file"], "steps:", len(run["composition"]), "elements:", run["elements"])
json.dump({"file": run["file"], "elements": run["elements"], "composition": run["composition"]}, open(dst, "w"))
