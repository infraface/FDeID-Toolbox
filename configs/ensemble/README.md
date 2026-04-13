# Ensemble Configuration Files

Configuration files for the model ensemble experiments (Table: Model ensemble results on LFW).

## Directory Structure

```
configs/ensemble/
├── baselines/              # Single method baselines
│   ├── ciagan.yaml
│   ├── tidim.yaml
│   ├── ksame_select.yaml
│   └── deid_rppg.yaml
├── sequential/             # Sequential ensemble (pipeline)
│   ├── ciagan_then_tidim.yaml
│   ├── ksame_select_then_tidim.yaml
│   └── deid_rppg_then_ciagan.yaml
├── parallel/               # Parallel ensemble (weighted fusion)
│   ├── ciagan_ksame_0.7_0.3.yaml
│   ├── ciagan_deid_rppg_0.5_0.5.yaml
│   └── ksame_tidim_0.6_0.4.yaml
└── attribute_guided/       # Attribute-guided ensemble
    ├── preserve_gender_expr.yaml
    ├── preserve_age_rppg.yaml
    └── preserve_landmark.yaml
```

## Usage Examples

### Run Baseline
```bash
python scripts/run_generative_deid.py --dataset lfw --method ciagan --save_dir output/
```

### Run Sequential Ensemble
```bash
python scripts/run_ensemble.py \
    --ensemble_mode sequential \
    --methods_config configs/ensemble/sequential/ciagan_then_tidim.yaml \
    --dataset lfw \
    --save_dir output/sequential_ciagan_tidim/
```

### Run Parallel Ensemble
```bash
python scripts/run_ensemble.py \
    --ensemble_mode parallel \
    --methods_config configs/ensemble/parallel/ciagan_ksame_0.7_0.3.yaml \
    --weights 0.7,0.3 \
    --dataset lfw \
    --save_dir output/parallel_ciagan_ksame/
```

### Run Attribute-Guided Ensemble
```bash
python scripts/run_ensemble.py \
    --ensemble_mode attribute_guided \
    --methods_config configs/ensemble/attribute_guided/preserve_gender_expr.yaml \
    --preserve gender,emotion \
    --suppress identity \
    --dataset lfw \
    --save_dir output/attr_guided_gender_expr/
```

## SLURM Job Submission

### Sequential: CIAGAN -> TI-DIM
```bash
sbatch jobs/run_ensemble.sh sequential lfw configs/ensemble/sequential/ciagan_then_tidim.yaml
```

### Parallel: 0.7*CIAGAN + 0.3*k-Same-Select
```bash
sbatch jobs/run_ensemble.sh parallel lfw configs/ensemble/parallel/ciagan_ksame_0.7_0.3.yaml "" "0.7,0.3"
```

### Attribute-Guided: preserve gender+expression
```bash
sbatch jobs/run_ensemble.sh attribute_guided lfw configs/ensemble/attribute_guided/preserve_gender_expr.yaml "" "" "gender,emotion" "identity"
```

## Configuration Format

Each YAML file contains a list of method configurations:

```yaml
- type: generative|adversarial|ksame|naive
  method_name: method_name
  description: Human-readable description
  param1: value1
  param2: value2
```

## Method Reference

| Type | method_name | Key Parameters |
|------|-------------|----------------|
| generative | ciagan | target_id (optional) |
| generative | deid_rppg | - |
| adversarial | tidim | epsilon, num_iter, decay_factor |
| ksame | select | k, selection_mode |

## Experiments for Paper Table

The configuration files map to the paper table experiments as follows:

| Table Row | Config File |
|-----------|-------------|
| CIAGAN | baselines/ciagan.yaml |
| TI-DIM (ε=8) | baselines/tidim.yaml |
| k-Same-Select (k=5) | baselines/ksame_select.yaml |
| DeID-rPPG | baselines/deid_rppg.yaml |
| CIAGAN → TI-DIM | sequential/ciagan_then_tidim.yaml |
| k-Same-Select → TI-DIM | sequential/ksame_select_then_tidim.yaml |
| DeID-rPPG → CIAGAN | sequential/deid_rppg_then_ciagan.yaml |
| 0.7×CIAGAN + 0.3×k-Same-Select | parallel/ciagan_ksame_0.7_0.3.yaml |
| 0.5×CIAGAN + 0.5×DeID-rPPG | parallel/ciagan_deid_rppg_0.5_0.5.yaml |
| 0.6×k-Same-Select + 0.4×TI-DIM | parallel/ksame_tidim_0.6_0.4.yaml |
| preserve={gender,expr} | attribute_guided/preserve_gender_expr.yaml |
| preserve={age,rPPG} | attribute_guided/preserve_age_rppg.yaml |
| preserve={landmark} | attribute_guided/preserve_landmark.yaml |
