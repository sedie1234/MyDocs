#!/usr/bin/env python3
"""
HW Feasibility Analysis Calculator

Calculates Queue, UB, and SM feasibility for YOLOv10 models
on Infetron-V2 NPU hardware.

Supports two execution models:
  - "parallel": all cores share tiles of every layer (baseline)
  - "sequential": cores form a pipeline, each core handles assigned layers

Usage:
    python calculate.py                    # Use default config.json
    python calculate.py --config my.json   # Use custom config
"""

import argparse
import csv
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Non-conv ops distribution by resolution
# ---------------------------------------------------------------------------
NON_CONV_OPS_BY_RES = {
    320: 0,
    160: 1,
    80: 4,
    40: 10,
    20: 15,
}


def get_non_conv_ops_for_resolution(out_h):
    if out_h >= 320:
        return NON_CONV_OPS_BY_RES[320]
    elif out_h >= 160:
        return NON_CONV_OPS_BY_RES[160]
    elif out_h >= 80:
        return NON_CONV_OPS_BY_RES[80]
    elif out_h >= 40:
        return NON_CONV_OPS_BY_RES[40]
    else:
        return NON_CONV_OPS_BY_RES[20]


# ---------------------------------------------------------------------------
# Tile size computation
# ---------------------------------------------------------------------------

def compute_ub_need(layer, T, metadata):
    """Compute total UB bytes needed for a tile of size T."""
    K_h = layer["kernel_h"]
    K_w = layer["kernel_w"]
    S = layer["stride"]
    C_in = layer["input_c"]
    C_out = layer["output_c"]
    ltype = layer["type"]
    weight_bytes = layer["weight_bytes"]

    if ltype == "1x1":
        if S == 1:
            input_tile = T * T * C_in
        else:
            input_tile = (S * T) * (S * T) * C_in
        output_tile = T * T * C_out
    else:  # Conv or DW
        in_h = S * T + K_h - S
        in_w = S * T + K_w - S
        input_tile = in_h * in_w * C_in
        output_tile = T * T * C_out

    total = input_tile + output_tile + weight_bytes + metadata
    return total, input_tile, output_tile


def find_adaptive_tile(layer, budget, metadata):
    """Find max tile size T where UB need <= budget. Binary search."""
    out_h = layer["output_h"]
    out_w = layer["output_w"]
    max_T = max(out_h, out_w)

    need_1, _, _ = compute_ub_need(layer, 1, metadata)
    if need_1 > budget:
        return 1, need_1, False  # UB_FAIL

    lo, hi = 1, max_T
    best_T = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        need, _, _ = compute_ub_need(layer, mid, metadata)
        if need <= budget:
            best_T = mid
            lo = mid + 1
        else:
            hi = mid - 1

    best_need, _, _ = compute_ub_need(layer, best_T, metadata)
    return best_T, best_need, True


def tiles_for_layer(layer, T):
    out_h = layer["output_h"]
    out_w = layer["output_w"]
    return math.ceil(out_h / T) * math.ceil(out_w / T)


def get_tile_size_for_layer(layer, tiling_case, metadata):
    """Returns (T, fits_in_budget) for a layer given tiling case."""
    if tiling_case["type"] == "fixed":
        T = min(tiling_case["tile_size"], layer["output_h"], layer["output_w"])
        return T, True
    else:
        budget = tiling_case["budget_bytes"]
        T, _, fits = find_adaptive_tile(layer, budget, metadata)
        return T, fits


# ---------------------------------------------------------------------------
# Sequential core assignment
# ---------------------------------------------------------------------------

def assign_layers_to_cores_sequential(layers, non_conv_ops_total, cores):
    """
    Assign layers to cores sequentially (pipeline model).
    Core 0 gets first N layers, Core 1 gets next N, etc.
    Also distributes non-conv ops proportionally.
    Returns list of (core_id, assigned_conv_layers, assigned_non_conv_count).
    """
    n = len(layers)
    base = n // cores
    remainder = n % cores

    assignments = []
    idx = 0
    for c in range(cores):
        count = base + (1 if c < remainder else 0)
        assigned = layers[idx:idx+count]
        idx += count

        # Non-conv ops: distribute based on resolutions of assigned layers
        nc = 0
        seen_res = set()
        for l in assigned:
            res = l["output_h"]
            if res not in seen_res:
                seen_res.add(res)
                nc += get_non_conv_ops_for_resolution(res)

        assignments.append({
            "core_id": c,
            "layers": assigned,
            "layer_indices": [l["index"] for l in assigned],
            "non_conv_ops": nc,
        })

    return assignments


# ---------------------------------------------------------------------------
# Queue analysis
# ---------------------------------------------------------------------------

def queue_analysis_parallel(layers, tiling_case, config, non_conv_ops):
    """Original parallel model: total_records / cores * margin."""
    metadata = config["policies"]["metadata_overhead_bytes"]
    cores = config["cores"]
    margin = config["margin_factor"]

    total_tile_ops = 0
    per_layer_tiles = []
    for layer in layers:
        T, _ = get_tile_size_for_layer(layer, tiling_case, metadata)
        n_tiles = tiles_for_layer(layer, T)
        per_layer_tiles.append(n_tiles)
        total_tile_ops += n_tiles

    non_conv_tile_ops = _calc_non_conv_tile_ops(layers, tiling_case, metadata)
    total_tile_ops += non_conv_tile_ops

    total_layers_with_nonconv = len(layers) + non_conv_ops
    instruction_records = total_tile_ops + total_layers_with_nonconv * 2 + 2

    per_core = instruction_records / cores * margin
    verdict = "PASS" if per_core <= config["hardware"]["q_max_records"] else "FAIL"

    return {
        "mode": "parallel",
        "conv_tile_ops": total_tile_ops - non_conv_tile_ops,
        "non_conv_tile_ops": non_conv_tile_ops,
        "total_tile_ops": total_tile_ops,
        "instruction_records": instruction_records,
        "per_core_avg": round(per_core, 1),
        "per_core_worst": round(per_core, 1),
        "worst_core_id": "N/A (avg)",
        "q_max_records": config["hardware"]["q_max_records"],
        "verdict": verdict,
        "core_details": None,
    }


def queue_analysis_sequential(layers, tiling_case, config, non_conv_ops):
    """Sequential pipeline model: each core handles assigned layers."""
    metadata = config["policies"]["metadata_overhead_bytes"]
    cores = config["cores"]
    margin = config["margin_factor"]

    assignments = assign_layers_to_cores_sequential(layers, non_conv_ops, cores)

    total_tile_ops_all = 0
    core_details = []

    for a in assignments:
        core_tile_ops = 0
        core_layer_tiles = []

        for layer in a["layers"]:
            T, _ = get_tile_size_for_layer(layer, tiling_case, metadata)
            n_tiles = tiles_for_layer(layer, T)
            core_tile_ops += n_tiles
            core_layer_tiles.append({
                "index": layer["index"],
                "name": layer["name"],
                "out_hw": f"{layer['output_h']}x{layer['output_w']}",
                "tile_size": T,
                "tiles": n_tiles,
            })

        # Non-conv ops for this core's resolutions
        nc_tile_ops = 0
        seen_res = set()
        for layer in a["layers"]:
            res = layer["output_h"]
            if res not in seen_res:
                seen_res.add(res)
                nc_count = get_non_conv_ops_for_resolution(res)
                if nc_count > 0:
                    T_nc, _ = get_tile_size_for_layer(layer, tiling_case, metadata)
                    nc_tile_ops += nc_count * tiles_for_layer(layer, T_nc)

        core_tile_ops += nc_tile_ops
        total_tile_ops_all += core_tile_ops

        num_core_layers = len(a["layers"]) + a["non_conv_ops"]
        core_records = core_tile_ops + num_core_layers * 2 + 1  # +1 for EPOCH or NOP

        core_details.append({
            "core_id": a["core_id"],
            "num_conv_layers": len(a["layers"]),
            "num_non_conv_ops": a["non_conv_ops"],
            "layer_range": f"{a['layer_indices'][0]}-{a['layer_indices'][-1]}" if a["layer_indices"] else "none",
            "conv_tile_ops": core_tile_ops - nc_tile_ops,
            "non_conv_tile_ops": nc_tile_ops,
            "total_tile_ops": core_tile_ops,
            "instruction_records": core_records,
            "per_core_with_margin": round(core_records * margin, 1),
            "layer_tiles": core_layer_tiles,
        })

    worst_core = max(core_details, key=lambda c: c["instruction_records"])
    worst_with_margin = round(worst_core["instruction_records"] * margin, 1)
    avg_records = sum(c["instruction_records"] for c in core_details) / cores
    q_max = config["hardware"]["q_max_records"]

    verdict = "PASS" if worst_core["instruction_records"] <= q_max else "FAIL"

    return {
        "mode": "sequential",
        "total_tile_ops": total_tile_ops_all,
        "instruction_records_total": sum(c["instruction_records"] for c in core_details),
        "per_core_avg": round(avg_records, 1),
        "per_core_worst": worst_core["instruction_records"],
        "per_core_worst_with_margin": worst_with_margin,
        "worst_core_id": worst_core["core_id"],
        "worst_core_layers": worst_core["layer_range"],
        "q_max_records": q_max,
        "verdict": verdict,
        "core_details": core_details,
    }


def _calc_non_conv_tile_ops(layers, tiling_case, metadata):
    non_conv_tile_ops = 0
    for res, count in NON_CONV_OPS_BY_RES.items():
        if count == 0:
            continue
        rep = None
        for l in layers:
            if l["output_h"] == res:
                rep = l
                break
        if rep is None:
            for l in layers:
                if l["output_h"] <= res:
                    rep = l
                    break
        if rep is None:
            rep = layers[-1]
        T, _ = get_tile_size_for_layer(rep, tiling_case, metadata)
        non_conv_tile_ops += count * tiles_for_layer(rep, T)
    return non_conv_tile_ops


# ---------------------------------------------------------------------------
# UB analysis — weight-first model
# Weight is FULLY loaded into UB first. Input is sliced to fit remaining space.
# ---------------------------------------------------------------------------

def compute_input_tile_hw(layer, T_out):
    """Given output tile T_out, compute input tile spatial dimensions."""
    K_h = layer["kernel_h"]
    K_w = layer["kernel_w"]
    S = layer["stride"]
    ltype = layer["type"]

    if ltype == "1x1":
        if S == 1:
            return T_out, T_out
        else:
            return S * T_out, S * T_out
    else:
        return S * T_out + K_h - S, S * T_out + K_w - S


def ub_analysis(layers, tiling_case, config):
    metadata = config["policies"]["metadata_overhead_bytes"]
    ub_size = config["hardware"]["ub_size_bytes"]
    results = []

    for layer in layers:
        weight = layer["weight_bytes"]
        # Step 1: Weight goes in first (fully, non-negotiable)
        remaining_for_tiles = ub_size - weight - metadata

        # Step 2: Weight alone exceeds UB?
        weight_fits = remaining_for_tiles > 0

        if tiling_case["type"] == "fixed":
            T = min(tiling_case["tile_size"], layer["output_h"], layer["output_w"])
            total, in_tile, out_tile = compute_ub_need(layer, T, metadata)
            budget = ub_size
            fits = total <= budget
        else:
            budget = tiling_case["budget_bytes"]
            T, total, fits = find_adaptive_tile(layer, budget, metadata)
            total, in_tile, out_tile = compute_ub_need(layer, T, metadata)
            fits = total <= budget

        # Input tile spatial dimensions
        in_tile_h, in_tile_w = compute_input_tile_hw(layer, T)

        # Fail reason
        if not weight_fits:
            fail_reason = "weight_exceeds_ub"
        elif not fits:
            fail_reason = "tile_too_large"
        else:
            fail_reason = ""

        results.append({
            "index": layer["index"],
            "name": layer["name"],
            "type": layer["type"],
            "kernel": f"{layer['kernel_h']}x{layer['kernel_w']}",
            "stride": layer["stride"],
            "input_h": layer["input_h"],
            "input_w": layer["input_w"],
            "input_c": layer["input_c"],
            "output_h": layer["output_h"],
            "output_w": layer["output_w"],
            "output_c": layer["output_c"],
            # UB breakdown: weight-first
            "weight_bytes": weight,
            "metadata_bytes": metadata,
            "remaining_for_tiles": max(remaining_for_tiles, 0),
            "weight_fits_ub": weight_fits,
            # Input slice (what gets cut)
            "input_tile_h": in_tile_h,
            "input_tile_w": in_tile_w,
            "input_tile_bytes": in_tile,
            # Output slice (derived from input)
            "output_tile_h": T,
            "output_tile_w": T,
            "output_tile_bytes": out_tile,
            # Totals
            "tile_io_bytes": in_tile + out_tile,
            "total_ub_need": total,
            "ub_budget": budget if tiling_case["type"] != "fixed" else ub_size,
            "fail_reason": fail_reason,
            "verdict": "PASS" if fits else "FAIL",
        })

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")
    fail_layers = [r for r in results if r["verdict"] == "FAIL"]
    weight_exceed = sum(1 for r in results if r["fail_reason"] == "weight_exceeds_ub")

    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "weight_exceed_count": weight_exceed,
        "total_layers": len(layers),
        "fail_layers": fail_layers,
        "all_layers": results,
        "verdict": "PASS" if fail_count == 0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# SM analysis — supports overlap mode
# ---------------------------------------------------------------------------

def sm_analysis(layers, config):
    sm_size = config["hardware"]["sm_size_bytes"]
    total_weight = sum(l["weight_bytes"] for l in layers)
    input_size = config["policies"]["input_size_bytes"]
    sm_overlap = config["policies"].get("sm_input_output_overlap", False)

    static_total = total_weight + input_size
    static_verdict = "PASS" if static_total <= sm_size else "FAIL"

    peak_results = []
    for layer in layers:
        in_fm = layer["input_bytes"]
        out_fm = layer["output_bytes"]

        if sm_overlap:
            # Input and output share SM space: only max(in, out) needed
            fm_need = max(in_fm, out_fm)
            peak = total_weight + fm_need
        else:
            # Both must coexist
            fm_need = in_fm + out_fm
            peak = total_weight + in_fm + out_fm

        fits = peak <= sm_size

        # Halo buffer calculation for overlap mode
        # When output overwrites input in SM, we need to pre-copy halo
        # rows into core (UB) so adjacent tiles can still read them.
        # Halo = (K-1) rows × input_full_width × C_in
        K = layer["kernel_h"]
        S = layer["stride"]
        halo_rows = K - 1  # rows of input overlap between adjacent tile rows
        input_w = layer["input_w"]
        C_in = layer["input_c"]

        if halo_rows > 0 and sm_overlap:
            # Bottom halo: full-width strip that next tile-row needs
            halo_bottom = halo_rows * input_w * C_in
            # Right halo: per-tile column overlap (smaller)
            # For simplicity: (K-1) columns × input_tile_height × C_in
            # Use conservative estimate with full input height per tile row
            in_tile_h = S * layer["output_h"] + K - S  # full input height (not tiled)
            halo_right = halo_rows * in_tile_h * C_in  # overestimate but safe
            halo_total = halo_bottom  # bottom halo is the binding constraint
        else:
            halo_bottom = 0
            halo_right = 0
            halo_total = 0

        peak_results.append({
            "index": layer["index"],
            "name": layer["name"],
            "input_fm_bytes": in_fm,
            "output_fm_bytes": out_fm,
            "fm_need_bytes": fm_need,
            "total_weight_bytes": total_weight,
            "peak_bytes": peak,
            "sm_size": sm_size,
            "overlap_mode": sm_overlap,
            "halo_rows": halo_rows,
            "halo_bottom_bytes": halo_bottom,
            "halo_total_bytes": halo_total,
            "verdict": "PASS" if fits else "FAIL",
        })

    peak_fail = [r for r in peak_results if r["verdict"] == "FAIL"]

    return {
        "total_weight_bytes": total_weight,
        "input_size_bytes": input_size,
        "static_total": static_total,
        "sm_size_bytes": sm_size,
        "static_verdict": static_verdict,
        "peak_pass_count": sum(1 for r in peak_results if r["verdict"] == "PASS"),
        "peak_fail_count": len(peak_fail),
        "peak_fail_layers": peak_fail,
        "peak_all_layers": peak_results,
    }


# ---------------------------------------------------------------------------
# Safe boundary analysis
# For each layer, find the weight/input constraints where both Q and UB pass.
# ---------------------------------------------------------------------------

def safe_boundary_analysis(layers, config, non_conv_ops):
    """
    For each core (sequential), compute:
    1. Q tile budget per core = q_max - overhead
    2. For each layer on that core: minimum T (output tile) to stay within Q budget
    3. From min T: max weight = UB - input_tile(T) - output_tile(T) - metadata
    4. Input tile size at that T
    """
    cores = config["cores"]
    q_max = config["hardware"]["q_max_records"]
    ub_size = config["hardware"]["ub_size_bytes"]
    metadata = config["policies"]["metadata_overhead_bytes"]

    assignments = assign_layers_to_cores_sequential(layers, non_conv_ops, cores)

    all_layer_bounds = {}  # index -> bounds info

    for a in assignments:
        core_id = a["core_id"]
        core_layers = a["layers"]
        nc_ops = a["non_conv_ops"]
        num_total_layers = len(core_layers) + nc_ops

        # Q budget for this core: total allowed tile-ops
        # records = tile_ops + num_layers*2 + 1 ≤ q_max
        q_tile_budget = q_max - num_total_layers * 2 - 1

        # Subtract non-conv tile-ops (estimate: nc_ops * avg tiles at their resolution)
        nc_tile_est = 0
        seen_res = set()
        for l in core_layers:
            res = l["output_h"]
            if res not in seen_res:
                seen_res.add(res)
                nc_count = get_non_conv_ops_for_resolution(res)
                if nc_count > 0:
                    # Assume non-conv uses same tile as conv at that resolution
                    # For safety estimate, use T=output_h (single tile)
                    nc_tile_est += nc_count * 1  # optimistic: 1 tile each

        conv_tile_budget = max(q_tile_budget - nc_tile_est, len(core_layers))

        # Distribute tile budget across layers proportional to output area
        total_output_area = sum(l["output_h"] * l["output_w"] for l in core_layers)
        if total_output_area == 0:
            total_output_area = 1

        for layer in core_layers:
            out_area = layer["output_h"] * layer["output_w"]
            # Layer's share of tile budget (proportional to output area)
            layer_tile_share = max(1, int(conv_tile_budget * out_area / total_output_area))

            # Find minimum T where tiles ≤ layer_tile_share
            out_h = layer["output_h"]
            out_w = layer["output_w"]

            # min T where ceil(H/T)*ceil(W/T) ≤ budget
            best_T = out_h  # start with full (1 tile)
            for T in range(1, max(out_h, out_w) + 1):
                tiles = math.ceil(out_h / T) * math.ceil(out_w / T)
                if tiles <= layer_tile_share:
                    best_T = T
                    break

            min_T_for_q = best_T
            tiles_at_min_T = tiles_for_layer(layer, min_T_for_q)

            # UB: at this min T, what's the max weight?
            _, in_tile_bytes, out_tile_bytes = compute_ub_need(layer, min_T_for_q, metadata)
            max_weight_for_ub = ub_size - in_tile_bytes - out_tile_bytes - metadata
            max_weight_for_ub = max(max_weight_for_ub, 0)

            # Input tile dimensions at min T
            in_h, in_w = compute_input_tile_hw(layer, min_T_for_q)

            # Current weight and check
            current_weight = layer["weight_bytes"]
            weight_ok = current_weight <= max_weight_for_ub

            # Also compute: for current weight, what's the adaptive T and resulting tiles?
            # (using UB full budget)
            T_adaptive, _, fits_ub = find_adaptive_tile(layer, ub_size, metadata)
            tiles_adaptive = tiles_for_layer(layer, T_adaptive)

            all_layer_bounds[layer["index"]] = {
                "index": layer["index"],
                "name": layer["name"],
                "type": layer["type"],
                "core_id": core_id,
                "output_h": out_h,
                "output_w": out_w,
                "output_c": layer["output_c"],
                "input_c": layer["input_c"],
                "kernel": f"{layer['kernel_h']}x{layer['kernel_w']}",
                "stride": layer["stride"],
                # Q constraints
                "q_tile_budget_core": q_tile_budget,
                "conv_tile_budget": conv_tile_budget,
                "layer_tile_share": layer_tile_share,
                "min_T_for_q": min_T_for_q,
                "tiles_at_min_T": tiles_at_min_T,
                # UB constraints at Q-safe T
                "in_tile_h_at_minT": in_h,
                "in_tile_w_at_minT": in_w,
                "in_tile_bytes_at_minT": in_tile_bytes,
                "out_tile_bytes_at_minT": out_tile_bytes,
                "max_weight_for_ub": max_weight_for_ub,
                # Current state
                "current_weight": current_weight,
                "weight_ok": weight_ok,
                "weight_margin": max_weight_for_ub - current_weight,
                # Adaptive Full (current weight)
                "T_adaptive_full": T_adaptive,
                "tiles_adaptive_full": tiles_adaptive,
                "fits_ub_full": fits_ub,
            }

    return all_layer_bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_json(config_path)
    model_data = load_json(os.path.join(script_dir, "model_data.json"))

    exec_model = config.get("execution_model", "parallel")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    summary_rows = []

    for model_key in config["models"]:
        mdata = model_data[model_key]
        layers = mdata["inference_layers"]
        non_conv_ops = mdata["non_conv_ops"]
        model_name = mdata["name"]

        all_results[model_key] = {"name": model_name, "tiling": {}}

        sm = sm_analysis(layers, config)
        all_results[model_key]["sm"] = sm

        for tc in config["tiling_cases"]:
            tc_name = tc["name"]

            if exec_model == "sequential":
                q = queue_analysis_sequential(layers, tc, config, non_conv_ops)
            else:
                q = queue_analysis_parallel(layers, tc, config, non_conv_ops)

            ub = ub_analysis(layers, tc, config)

            all_results[model_key]["tiling"][tc_name] = {
                "queue": q,
                "ub": ub,
            }

            # Overall verdict: all three must PASS
            q_v = q["verdict"]
            ub_v = ub["verdict"]
            sm_v = "FAIL" if sm["static_verdict"] == "FAIL" or sm["peak_fail_count"] > 0 else "PASS"
            overall = "PASS" if q_v == "PASS" and ub_v == "PASS" and sm_v == "PASS" else "FAIL"

            worst_q = q.get("per_core_worst", q.get("per_core_avg", 0))

            summary_rows.append([
                model_name, tc_name,
                q.get("total_tile_ops", ""),
                worst_q, config["hardware"]["q_max_records"], q_v,
                ub["pass_count"], ub["fail_count"], ub_v,
                sm_v, sm["peak_fail_count"],
                overall,
            ])

    # Save results
    save_json(os.path.join(results_dir, "results.json"), all_results)

    save_csv(
        os.path.join(results_dir, "feasibility_summary.csv"),
        ["Model", "Tiling", "TotalTileOps",
         "Q_WorstCore", "Q_Max", "Q_Verdict",
         "UB_Pass", "UB_Fail", "UB_Verdict",
         "SM_Verdict", "SM_PeakFail",
         "Overall"],
        summary_rows,
    )

    # Queue CSV
    q_rows = []
    for model_key in config["models"]:
        for tc in config["tiling_cases"]:
            q = all_results[model_key]["tiling"][tc["name"]]["queue"]
            if exec_model == "sequential" and q.get("core_details"):
                for cd in q["core_details"]:
                    q_rows.append([
                        model_data[model_key]["name"], tc["name"],
                        f"Core{cd['core_id']}", cd["layer_range"],
                        cd["num_conv_layers"], cd["num_non_conv_ops"],
                        cd["total_tile_ops"], cd["instruction_records"],
                        config["hardware"]["q_max_records"],
                        "PASS" if cd["instruction_records"] <= config["hardware"]["q_max_records"] else "FAIL",
                    ])
            else:
                q_rows.append([
                    model_data[model_key]["name"], tc["name"],
                    "All", "0-end",
                    len(model_data[model_key]["inference_layers"]),
                    model_data[model_key]["non_conv_ops"],
                    q.get("total_tile_ops", ""),
                    q.get("per_core_avg", ""),
                    config["hardware"]["q_max_records"],
                    q["verdict"],
                ])
    save_csv(
        os.path.join(results_dir, "queue_analysis.csv"),
        ["Model", "Tiling", "Core", "LayerRange",
         "ConvLayers", "NonConvOps",
         "TileOps", "InstrRecords", "Q_Max", "Verdict"],
        q_rows,
    )

    # UB CSV — weight-first breakdown
    ub_rows = []
    for model_key in config["models"]:
        for tc in config["tiling_cases"]:
            ub = all_results[model_key]["tiling"][tc["name"]]["ub"]
            for lr in ub["all_layers"]:
                ub_rows.append([
                    model_data[model_key]["name"], tc["name"],
                    lr["index"], lr["name"], lr["type"],
                    lr["kernel"], lr["stride"],
                    f'{lr["input_h"]}x{lr["input_w"]}x{lr["input_c"]}',
                    f'{lr["output_h"]}x{lr["output_w"]}x{lr["output_c"]}',
                    lr["weight_bytes"],
                    lr["remaining_for_tiles"],
                    f'{lr["input_tile_h"]}x{lr["input_tile_w"]}',
                    lr["input_tile_bytes"],
                    f'{lr["output_tile_h"]}x{lr["output_tile_w"]}',
                    lr["output_tile_bytes"],
                    lr["tile_io_bytes"],
                    lr["metadata_bytes"],
                    lr["total_ub_need"], lr["ub_budget"],
                    lr["weight_fits_ub"],
                    lr["fail_reason"],
                    lr["verdict"],
                ])
    save_csv(
        os.path.join(results_dir, "ub_analysis.csv"),
        ["Model", "Tiling", "Index", "Name", "Type",
         "Kernel", "Stride", "InputShape", "OutputShape",
         "Weight_B", "Remaining_for_tiles",
         "InTile_HW", "InTile_B", "OutTile_HW", "OutTile_B",
         "TileIO_B", "Meta_B",
         "TotalUB", "Budget", "WeightFitsUB", "FailReason", "Verdict"],
        ub_rows,
    )

    # SM CSV
    sm_rows = []
    for model_key in config["models"]:
        sm = all_results[model_key]["sm"]
        sm_rows.append([
            model_data[model_key]["name"], "Static",
            sm["total_weight_bytes"], sm["input_size_bytes"],
            sm["static_total"], sm["sm_size_bytes"], sm["static_verdict"],
        ])
        for lr in sm["peak_fail_layers"]:
            sm_rows.append([
                model_data[model_key]["name"],
                f'Peak_L{lr["index"]}',
                lr["total_weight_bytes"],
                lr["input_fm_bytes"] + lr["output_fm_bytes"],
                lr["peak_bytes"], lr["sm_size"], lr["verdict"],
            ])
    save_csv(
        os.path.join(results_dir, "sm_analysis.csv"),
        ["Model", "Check", "WeightBytes", "FMBytes", "Total", "SM_Size", "Verdict"],
        sm_rows,
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("=" * 90)
    print(f"HW Feasibility Analysis Results  [execution_model={exec_model}]")
    print("=" * 90)

    for model_key in config["models"]:
        mdata = model_data[model_key]
        model_name = mdata["name"]
        sm = all_results[model_key]["sm"]

        print(f"\n{'─' * 90}")
        print(f"  {model_name} ({len(mdata['inference_layers'])} conv layers + {mdata['non_conv_ops']} non-conv ops)")
        print(f"  Total weight: {sm['total_weight_bytes']:,} bytes ({sm['total_weight_bytes']/1024/1024:.2f} MB)")
        print(f"{'─' * 90}")

        # Queue
        q_max = config["hardware"]["q_max_records"]
        print(f"\n  Queue Analysis (Q max = {q_max} records, {config['cores']} cores, mode={exec_model}):")

        if exec_model == "sequential":
            print(f"  {'Tiling':<20} {'WorstCore':>10} {'WorstRec':>10} {'WorstLyrs':>12} {'AvgRec':>10} {'Verdict':>8}")
            print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
            for tc in config["tiling_cases"]:
                q = all_results[model_key]["tiling"][tc["name"]]["queue"]
                print(f"  {tc['name']:<20} {'Core'+str(q['worst_core_id']):>10} {q['per_core_worst']:>10} {q['worst_core_layers']:>12} {q['per_core_avg']:>10.1f} {q['verdict']:>8}")

            # Show per-core detail for worst tiling case
            worst_tc = max(config["tiling_cases"],
                          key=lambda tc: all_results[model_key]["tiling"][tc["name"]]["queue"]["per_core_worst"])
            q_worst = all_results[model_key]["tiling"][worst_tc["name"]]["queue"]
            print(f"\n  Per-core breakdown ({worst_tc['name']}):")
            print(f"  {'Core':>6} {'Layers':>12} {'ConvLyrs':>8} {'TileOps':>10} {'Records':>10} {'Verdict':>8}")
            print(f"  {'─'*6} {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
            for cd in q_worst["core_details"]:
                v = "PASS" if cd["instruction_records"] <= q_max else "FAIL"
                print(f"  {cd['core_id']:>6} {cd['layer_range']:>12} {cd['num_conv_layers']:>8} {cd['total_tile_ops']:>10} {cd['instruction_records']:>10} {v:>8}")
        else:
            print(f"  {'Tiling':<20} {'TileOps':>10} {'InstrRec':>10} {'PerCore':>10} {'Verdict':>8}")
            print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
            for tc in config["tiling_cases"]:
                q = all_results[model_key]["tiling"][tc["name"]]["queue"]
                print(f"  {tc['name']:<20} {q['total_tile_ops']:>10} {q['instruction_records']:>10} {q['per_core_avg']:>10.1f} {q['verdict']:>8}")

        # UB — weight-first breakdown
        ub_bytes = config['hardware']['ub_size_bytes']
        print(f"\n  UB Analysis (UB = {ub_bytes:,} bytes = {ub_bytes/1024:.0f} KiB, weight-first model):")
        print(f"  방식: weight 전량 적재 → 남은 공간에 input slice + output slice")
        print(f"  {'Tiling':<20} {'Pass':>6} {'Fail':>6} {'WtOver':>6} {'Verdict':>8}")
        print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        for tc in config["tiling_cases"]:
            ub = all_results[model_key]["tiling"][tc["name"]]["ub"]
            wt_over = ub.get("weight_exceed_count", 0)
            print(f"  {tc['name']:<20} {ub['pass_count']:>6} {ub['fail_count']:>6} {wt_over:>6} {ub['verdict']:>8}")
            if 0 < ub["fail_count"] <= 10:
                for fl in ub["fail_layers"]:
                    wt = fl['weight_bytes']
                    remaining = fl['remaining_for_tiles']
                    in_hw = f"{fl['input_tile_h']}x{fl['input_tile_w']}"
                    out_hw = f"{fl['output_tile_h']}x{fl['output_tile_w']}"
                    reason = fl.get('fail_reason', '')
                    if reason == "weight_exceeds_ub":
                        print(f"    FAIL: [{fl['index']}] {fl['name']} weight={wt:,} > UB={fl['ub_budget']:,} (weight 자체가 UB 초과)")
                    else:
                        print(f"    FAIL: [{fl['index']}] {fl['name']} weight={wt:,} + in_tile({in_hw})={fl['input_tile_bytes']:,} + out_tile({out_hw})={fl['output_tile_bytes']:,} = {fl['total_ub_need']:,} > {fl['ub_budget']:,}")

        # SM
        overlap_mode = config["policies"].get("sm_input_output_overlap", False)
        sm_v = "FAIL" if sm["static_verdict"] == "FAIL" or sm["peak_fail_count"] > 0 else "PASS"
        print(f"\n  SM Analysis (SM = {sm['sm_size_bytes']:,} bytes = {sm['sm_size_bytes']/1024/1024:.0f} MiB, overlap={overlap_mode}):")
        if overlap_mode:
            print(f"  Mode: input/output overlap (peak = weight + max(in_FM, out_FM))")
        else:
            print(f"  Mode: separate (peak = weight + in_FM + out_FM)")
        print(f"  Static (weight+input): {sm['static_total']:,} / {sm['sm_size_bytes']:,} -> {sm['static_verdict']}")
        if sm["peak_fail_count"] > 0:
            print(f"  Peak failures: {sm['peak_fail_count']} layers exceed SM during execution")
            for pf in sm["peak_fail_layers"][:5]:
                print(f"    [{pf['index']}] {pf['name']} peak={pf['peak_bytes']:,} (in={pf['input_fm_bytes']:,} out={pf['output_fm_bytes']:,})")
            if sm["peak_fail_count"] > 5:
                print(f"    ... and {sm['peak_fail_count'] - 5} more")
        else:
            print(f"  Peak: all layers PASS")
        # Show halo buffer requirements for overlap mode
        if overlap_mode:
            halo_layers = [r for r in sm["peak_all_layers"] if r["halo_total_bytes"] > 0]
            if halo_layers:
                max_halo = max(halo_layers, key=lambda x: x["halo_total_bytes"])
                print(f"\n  Halo buffer (core에 미리 복사해야 하는 input 양, overlap 시):")
                print(f"  {'Layer':>6} {'Name':<35} {'K':>3} {'Halo rows':>9} {'Halo bytes':>12} {'= KB':>8}")
                print(f"  {'─'*6} {'─'*35} {'─'*3} {'─'*9} {'─'*12} {'─'*8}")
                shown = set()
                for lr in sorted(halo_layers, key=lambda x: -x["halo_total_bytes"])[:15]:
                    key = (lr["halo_rows"], lr["halo_total_bytes"])
                    if key in shown and lr != max_halo:
                        continue
                    shown.add(key)
                    print(f"  {lr['index']:>6} {lr['name']:<35} {lr['halo_rows']+1:>3} {lr['halo_rows']:>9} {lr['halo_total_bytes']:>12,} {lr['halo_total_bytes']/1024:>7.1f}K")
                print(f"\n  Max halo buffer: [{max_halo['index']}] {max_halo['name']} = {max_halo['halo_total_bytes']:,} bytes ({max_halo['halo_total_bytes']/1024:.1f} KB)")
                print(f"  → UB에 추가로 이 만큼의 공간이 필요 (or 별도 halo buffer)")

    # Final summary table
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'─' * 90}")
    print(f"  {'Model':<12} {'Tiling':<18} {'Q':>8} {'UB':>8} {'SM':>8} {'Overall':>8}")
    print(f"  {'─'*12} {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for row in summary_rows:
        model, tiling = row[0], row[1]
        q_v, ub_v, sm_v, overall = row[5], row[8], row[9], row[11]
        print(f"  {model:<12} {tiling:<18} {q_v:>8} {ub_v:>8} {sm_v:>8} {overall:>8}")

    # Safe boundary analysis
    print(f"\n{'=' * 90}")
    print("  SAFE BOUNDARY ANALYSIS (Q+UB 동시 만족하는 weight/input 한계)")
    print(f"{'=' * 90}")

    for model_key in config["models"]:
        mdata = model_data[model_key]
        layers = mdata["inference_layers"]
        non_conv_ops = mdata["non_conv_ops"]
        model_name = mdata["name"]

        bounds = safe_boundary_analysis(layers, config, non_conv_ops)
        all_results[model_key]["safe_bounds"] = bounds

        print(f"\n  {model_name}:")
        print(f"  UB = {config['hardware']['ub_size_bytes']:,} bytes, Q = {config['hardware']['q_max_records']} records/core")
        print(f"\n  {'Idx':>4} {'Core':>4} {'Name':<32} {'Type':<5} {'OutHW':>8} {'CurWt':>8} {'MaxWt':>8} {'Margin':>8} {'MinT':>5} {'Tiles':>6} {'InTile':>10} {'OK':>4}")
        print(f"  {'─'*4} {'─'*4} {'─'*32} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*6} {'─'*10} {'─'*4}")

        tight_layers = []
        for idx in sorted(bounds.keys()):
            b = bounds[idx]
            cur_w = b["current_weight"]
            max_w = b["max_weight_for_ub"]
            margin = b["weight_margin"]
            ok = "OK" if b["weight_ok"] else "FAIL"
            in_hw = f"{b['in_tile_h_at_minT']}x{b['in_tile_w_at_minT']}"
            out_hw = f"{b['output_h']}x{b['output_w']}"

            # Only show layers where margin < 50% of current weight, or FAIL
            if not b["weight_ok"] or margin < cur_w * 0.5 or cur_w > 30000:
                print(f"  {idx:>4} {b['core_id']:>4} {b['name']:<32} {b['type']:<5} {out_hw:>8} {cur_w:>7,} {max_w:>7,} {margin:>+7,} {b['min_T_for_q']:>5} {b['tiles_at_min_T']:>6} {in_hw:>10} {ok:>4}")
                if not b["weight_ok"]:
                    tight_layers.append(b)

        if tight_layers:
            print(f"\n  ⚠ {len(tight_layers)} layers에서 현재 weight > 안전 상한:")
            for b in tight_layers:
                print(f"    [{b['index']}] {b['name']}: weight {b['current_weight']:,} > max {b['max_weight_for_ub']:,} (초과 {-b['weight_margin']:,} bytes)")
        else:
            print(f"\n  ✓ 모든 layers에서 weight가 안전 상한 이내")

        # Summary stats
        margins = [bounds[i]["weight_margin"] for i in bounds]
        min_margin = min(margins)
        min_margin_layer = [bounds[i] for i in bounds if bounds[i]["weight_margin"] == min_margin][0]
        print(f"\n  가장 빡빡한 layer: [{min_margin_layer['index']}] {min_margin_layer['name']}")
        print(f"    weight={min_margin_layer['current_weight']:,}, max={min_margin_layer['max_weight_for_ub']:,}, margin={min_margin:+,} bytes")

    # Save safe boundary CSV
    sb_rows = []
    for model_key in config["models"]:
        bounds = all_results[model_key].get("safe_bounds", {})
        for idx in sorted(bounds.keys()):
            b = bounds[idx]
            sb_rows.append([
                model_data[model_key]["name"],
                b["index"], b["core_id"], b["name"], b["type"],
                b["kernel"], b["stride"],
                f"{b['output_h']}x{b['output_w']}", b["output_c"], b["input_c"],
                b["current_weight"],
                b["max_weight_for_ub"],
                b["weight_margin"],
                b["weight_ok"],
                b["min_T_for_q"],
                b["tiles_at_min_T"],
                b["layer_tile_share"],
                f"{b['in_tile_h_at_minT']}x{b['in_tile_w_at_minT']}",
                b["in_tile_bytes_at_minT"],
                b["out_tile_bytes_at_minT"],
                b["T_adaptive_full"],
                b["tiles_adaptive_full"],
            ])
    save_csv(
        os.path.join(results_dir, "safe_boundary.csv"),
        ["Model", "Index", "Core", "Name", "Type",
         "Kernel", "Stride", "OutHW", "OutC", "InC",
         "CurWeight", "MaxWeight", "Margin", "WeightOK",
         "MinT_Q", "Tiles_Q", "TileShare",
         "InTile_HW", "InTile_B", "OutTile_B",
         "T_AdaptFull", "Tiles_AdaptFull"],
        sb_rows,
    )

    print(f"\n{'=' * 90}")
    print(f"Results saved to: {results_dir}/")
    print(f"  + safe_boundary.csv")
    print(f"{'=' * 90}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW Feasibility Analysis Calculator")
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.json")

    run(config_path)
