# HW Feasibility Analysis Calculator

## Usage
```bash
python calculate.py                    # Use default config.json
python calculate.py --config my.json   # Use custom config
```

## Files
- `config.json`: Analysis settings (edit this to change scenarios)
- `model_data.json`: Layer data (regenerate with parse_md.py if model changes)
- `calculate.py`: Calculation engine
- `parse_md.py`: One-time helper to parse conv layer analysis markdown into JSON
- `results/`: Output directory (created automatically)

## Output Files
- `results/results.json`: Complete results with all computed numbers
- `results/feasibility_summary.csv`: One-line-per-scenario summary
- `results/queue_analysis.csv`: Queue record counts per tiling strategy
- `results/ub_analysis.csv`: Per-layer UB pass/fail for all tiling strategies
- `results/sm_analysis.csv`: SM static and peak analysis

## Changing Settings
Edit `config.json`, then re-run. Key fields:
- `tiling_cases`: add/remove/modify tiling strategies
- `hardware.sm_size_bytes`: change SM size
- `hardware.ub_size_bytes`: change UB size
- `policies.weight_in_ub`: toggle weight streaming
- `policies.metadata_overhead_bytes`: per-tile metadata overhead
- `margin_factor`: adjust safety margin for queue analysis
- `cores`: number of NPU cores

## Model Data
`model_data.json` contains inference-only layers:
- v10n: 83 layers (backbone+neck 0-58, one2one 83-106, renumbered 0-82)
- v10s: 87 layers (backbone+neck 0-62, one2one 87-110, renumbered 0-86)

To regenerate from the analysis markdown:
```bash
python parse_md.py
```

## Analysis Methodology

### Queue Analysis
- Counts total tile operations across all layers
- Adds LAYER_START/LAYER_END per layer + EPOCH_COMMIT + STOP
- Divides by cores, multiplies by safety margin
- Compares against Q max records (512)

### UB Analysis  
- Fixed tiling: checks if input_tile + output_tile + weight + metadata fits in UB
- Adaptive tiling: binary searches for largest tile that fits in budget
- Reports pass/fail per layer

### SM Analysis
- Static: checks if total_weight + input_image fits in SM
- Peak: checks per-layer if total_weight + input_FM + output_FM fits in SM
