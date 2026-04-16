#!/usr/bin/env python3
"""One-time helper: parse the conv layer analysis markdown and emit model_data.json."""
import re, json, math

SRC = "/home/keti/workspace/AICompiler/docs/agent_results/etc/20260401_yolov10_conv_layer_analysis.md"

def parse_bytes(s):
    """Parse '1.17 MB', '4.5 KB', '432 B' etc. to exact integer bytes."""
    s = s.strip()
    m = re.match(r'([\d.]+)\s*(MB|KB|B)', s)
    if not m:
        raise ValueError(f"Cannot parse bytes: {s!r}")
    val = float(m.group(1))
    unit = m.group(2)
    if unit == 'MB':
        return int(round(val * 1024 * 1024))
    elif unit == 'KB':
        return int(round(val * 1024))
    else:
        return int(round(val))

def parse_shape(s):
    """Parse '(1,640,640,3)' -> tuple of ints."""
    s = s.strip().strip('()')
    return tuple(int(x) for x in s.split(','))

def parse_weight_shape(s):
    """Parse '(3,3,3,16)' -> (Kh, Kw, Cin_per_group, Cout)"""
    return parse_shape(s)

def parse_table(lines):
    """Parse markdown table rows into layer dicts."""
    layers = []
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]  # remove empty from leading/trailing |
        if len(cells) < 10:
            continue
        # Skip header/separator
        if cells[0].startswith('#') or cells[0].startswith('-') or cells[0] == '---:' or '---' in cells[0]:
            continue
        try:
            idx = int(cells[0])
        except ValueError:
            continue

        name = cells[1].strip('`')
        ltype = cells[2]
        groups = int(cells[3])
        stride = int(cells[4])

        inp_shape = parse_shape(cells[5])
        # Compute exact bytes from shape (N*H*W*C * 1 byte per u8/s8)
        inp_bytes = inp_shape[0] * inp_shape[1] * inp_shape[2] * inp_shape[3]
        wt_shape = parse_weight_shape(cells[7])
        # Compute exact weight bytes from shape (Kh*Kw*Cin_per_group*Cout)
        wt_bytes = wt_shape[0] * wt_shape[1] * wt_shape[2] * wt_shape[3]
        out_shape = parse_shape(cells[9])
        out_bytes = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]

        layers.append({
            "index": idx,
            "name": name,
            "type": ltype,
            "groups": groups,
            "stride": stride,
            "kernel_h": wt_shape[0],
            "kernel_w": wt_shape[1],
            "input_h": inp_shape[1],
            "input_w": inp_shape[2],
            "input_c": inp_shape[3],
            "output_h": out_shape[1],
            "output_w": out_shape[2],
            "output_c": out_shape[3],
            "weight_bytes": wt_bytes,
            "input_bytes": inp_bytes,
            "output_bytes": out_bytes
        })
    return layers

with open(SRC) as f:
    content = f.read()

# Split into v10n and v10s sections
v10n_start = content.index("## YOLOv10n")
v10s_start = content.index("## YOLOv10s")
comparison_start = content.index("## YOLOv10n vs YOLOv10s")

v10n_section = content[v10n_start:v10s_start]
v10s_section = content[v10s_start:comparison_start]

v10n_all = parse_table(v10n_section.split('\n'))
v10s_all = parse_table(v10s_section.split('\n'))

print(f"v10n total parsed: {len(v10n_all)} layers (indices {v10n_all[0]['index']}-{v10n_all[-1]['index']})")
print(f"v10s total parsed: {len(v10s_all)} layers (indices {v10s_all[0]['index']}-{v10s_all[-1]['index']})")

# v10n inference: 0-58 (backbone+neck) + 83-106 (one2one)
# But looking at the data: indices 0-58 are backbone+neck (59 layers)
# indices 59-82 are one2many detect head (cv2/cv3 without "one2one")
# indices 83-106 are one2one detect head
# Wait, let me check: the note says "layers 0-58 (backbone+neck) + layers 83-106 (one2one)"
# That means indices 0-58 = 59 layers, indices 83-106 = 24 layers, total 83

def is_one2one(layer):
    return "one2one" in layer["name"]

def is_backbone_neck(layer, model):
    """Backbone+neck layers end before the detect head starts."""
    # For v10n: 0-58 backbone+neck
    # For v10s: 0-62 backbone+neck
    # Detect head layers contain "model.23." prefix
    return "model.23." not in layer["name"]

# v10n: backbone+neck = not model.23, one2one = has one2one in name
v10n_inference = []
for l in v10n_all:
    if "model.23." not in l["name"] or "one2one" in l["name"]:
        v10n_inference.append(l)

v10s_inference = []
for l in v10s_all:
    if "model.23." not in l["name"] or "one2one" in l["name"]:
        v10s_inference.append(l)

# Renumber
for i, l in enumerate(v10n_inference):
    l["index"] = i
for i, l in enumerate(v10s_inference):
    l["index"] = i

print(f"v10n inference: {len(v10n_inference)} layers")
print(f"v10s inference: {len(v10s_inference)} layers")

v10n_total_weight = sum(l["weight_bytes"] for l in v10n_inference)
v10s_total_weight = sum(l["weight_bytes"] for l in v10s_inference)
print(f"v10n inference weight: {v10n_total_weight} bytes ({v10n_total_weight/1024/1024:.2f} MB)")
print(f"v10s inference weight: {v10s_total_weight} bytes ({v10s_total_weight/1024/1024:.2f} MB)")

model_data = {
    "v10n": {
        "name": "YOLOv10n",
        "inference_layers": v10n_inference,
        "total_weight_bytes": v10n_total_weight,
        "non_conv_ops": 30
    },
    "v10s": {
        "name": "YOLOv10s",
        "inference_layers": v10s_inference,
        "total_weight_bytes": v10s_total_weight,
        "non_conv_ops": 30
    }
}

out_path = "/home/keti/workspace/AICompiler/docs/MyDocs/20260401_HW_Feasibility_Analysis/tools/model_data.json"
with open(out_path, 'w') as f:
    json.dump(model_data, f, indent=2)

print(f"\nWritten to {out_path}")
