# Comparison Algorithms Setup & Execution Guide

This directory contains scripts and configurations for running **BioHEL**, **BioHEL RPE**, and **RIPPER** algorithms on benchmark datasets for comparative evaluation against HEROS.

## Overview

This experimental pipeline enables:
- Cross-validation partitioned execution across multiple random seeds
- BioHEL training with optional RPE (Rule Post-processing Engine) extension
- RIPPER rule learning via the Wittgenstein Python package
- Automated result aggregation and statistical analysis
- HPC (High-Performance Computing) batch job submission and management
- Reproducible paper results with all commands documented in `run_commands.txt`

## Algorithm Implementations

### BioHEL (Bioinformatic Hierarchical Evolutionary Learning)

**BioHEL** is an evolutionary algorithm-based learning classifier system optimized for bioinformatics applications but applicable to general classification tasks.

#### Installation & Setup

##### 1. Download BioHEL Source

Visit: **https://ico2s.org/software/biohel.html**

- Download the BioHEL source code package
- Extract the archive:
  ```bash
  tar -xzf biohel_source.tar.gz
  cd biohel_source
  ```

##### 2. Build C Binaries

Follow the build instructions provided with BioHEL source (typically):

```bash
# Check for build requirements (see INSTALL or README in source)
./configure
make
make install
```

**Common build dependencies** (adjust for your system):
```bash
# macOS
brew install gcc

# Ubuntu/Debian
sudo apt-get install build-essential
```

After successful build, the `biohel` executable will be generated.

##### 3. Install in Comparison Scripts Directory

Once built, make the binary accessible to the comparison scripts:

```bash
# Option A: Copy the binary to the comparison scripts directory
cp /path/to/built/biohel ./biohel

# Option B: Create a symlink
ln -s /path/to/built/biohel ./biohel

# Option C: Add binary location to your PATH
export PATH="/path/to/built:$PATH"
```

Verify installation:
```bash
./biohel --help
# or
which biohel
```

#### BioHEL Configuration

Configuration is managed through `.conf` files. See `biohel_sample.conf` for an example with key parameters:

- **Population parameters**: `pop size`, `prob crossover`, `prob individual mutation`
- **Fitness function**: MDL (Minimum Description Length) based
- **Hyperrectangle representation**: Used for rule generalization
- **GPU support**: CUDA configuration options available
- **Iterations & coverage**: Controlled via `iterations`, `coverage ratio`

Modify configuration parameters in `.conf` files as needed for your experiments.

### BioHEL RPE (Rule Post-processing Engine)

BioHEL RPE provides post-training rule refinement and optimization.

#### Installation & Setup

##### 1. Download BioHEL RPE Source

Visit: **http://ico2s.org/software/biohel-rpe.html**

- Download the BioHEL RPE source code package
- Extract the archive:
  ```bash
  tar -xzf biohel_rpe_source.tar.gz
  cd biohel_rpe_source
  ```

##### 2. Build RPE Binary

Follow the build instructions provided with BioHEL RPE source:

```bash
./configure
make
make install
```

##### 3. Install in Comparison Scripts Directory

Make the `postprocess` executable accessible:

```bash
# Option A: Copy the binary
cp /path/to/built/postprocess ./postprocess

# Option B: Create a symlink
ln -s /path/to/built/postprocess ./postprocess

# Option C: Add to PATH
export PATH="/path/to/built:$PATH"
```

Verify installation:
```bash
./postprocess --help
# or
which postprocess
```

#### Using BioHEL RPE

Enable RPE post-processing in job scripts with the `--enable_rpe` flag:

```bash
python job_biohel_hpc.py \
  --d <training_data_file> \
  --o <output_directory> \
  --enable_rpe
```

RPE refines discovered rules for improved interpretability and performance. Configuration is managed through `postprocess.conf`.

### RIPPER (Repeated Incremental Pruning to Produce Error Reduction)

**RIPPER** is a classic rule-learning algorithm implemented via the **Wittgenstein** Python package.

#### Installation

Install the Wittgenstein package using pip:

```bash
pip install wittgenstein
```

Verify installation:
```bash
python -c "import wittgenstein; print(wittgenstein.__version__)"
```

This provides a Python interface to RIPPER, enabling rule-based classification with minimal overhead and cross-platform compatibility.

#### RIPPER Configuration

RIPPER is configured through command-line arguments in job scripts:
- `--verbosity`: Control output verbosity (0 = minimal, higher = verbose)
- Dataset labels: outcome label, instance ID, excluded columns
- Random seed management for reproducibility

No additional binary installation required—Wittgenstein handles all RIPPER functionality in Python.

## Directory Structure

```
comparison_algs_scripts/
├── README.md                      # This file
├── run_commands.txt               # All commands needed to reproduce paper results
├── biohel_sample.conf             # Example BioHEL configuration file
├── postprocess.conf               # BioHEL RPE configuration
│
├── Executables (must be built and placed here):
├── biohel                          # BioHEL binary (built from source)
├── postprocess                    # BioHEL RPE binary (built from source)
│
├── Job submission scripts (HPC LSF):
├── job_biohel_hpc.py              # Single BioHEL job (single dataset fold)
├── job_ripper_hpc.py              # Single RIPPER job (single dataset fold)
├── job_biohel_sum_hpc.py          # BioHEL aggregation job
├── job_ripper_sum_hpc.py          # RIPPER aggregation job
├── job_sum_table_hpc.py           # Final results table generation
│
├── Runner scripts (generate & submit batch jobs):
├── run_biohel_hpc.py              # Submit BioHEL jobs across CV folds & seeds
├── run_ripper_hpc.py              # Submit RIPPER jobs across CV folds & seeds
├── run_biohel_sum_hpc.py          # Submit BioHEL summary job
├── run_ripper_sum_hpc.py          # Submit RIPPER summary job
│
├── Utilities:
├── cleanup_hpc_artifacts.py       # Remove scratch files and logs
├── check_failed_jobs.py           # Monitor job completion status
│
└── stats_scripts/                 # Statistical comparison scripts
    ├── combine_heros_for_stats.py # Aggregate HEROS results
    └── job_statistical_test.py    # Wilcoxon signed-rank tests
```

## Prerequisites

Before running experiments, ensure all of the following are installed and configured:

### 1. Python Environment
- **Python 3.7+** installed
- Required Python package:
  ```bash
  pip install wittgenstein
  ```

### 2. BioHEL Binary
- Build from source: https://ico2s.org/software/biohel.html
- Place executable as `./biohel` or add to PATH
- Verify with: `./biohel --help`

### 3. BioHEL RPE Binary (Optional, for RPE post-processing)
- Build from source: http://ico2s.org/software/biohel-rpe.html
- Place executable as `./postprocess` or add to PATH
- Verify with: `./postprocess --help`

### 4. Data Preparation
- Input data should be in tab-separated format (`.txt`)
- Cross-validation partitions should be created first using the CV partitioning script
- Dataset must have: outcome/class column, instance ID column, optionally excluded columns

### 5. HPC System Access (if running on cluster)
- LSF (Load Sharing Facility) or compatible job scheduler
- Access to shared storage for data and output

## Workflow: From Data to Results

### Step 1: Run BioHEL Training

#### Option A: Single Debug Run (Local, No HPC)

For testing on a single fold without HPC submission:

```bash
python job_biohel_hpc.py \
  --d <training_data_file> \
  --o <output_directory> \
  --ol <outcome_label> \
  --il <instance_id_label> \
  --el <excluded_label> \
  --rs <random_seed> \
  --enable_rpe \
  --v
```

#### Option B: Full Batch Submission (HPC/LSF)

Submit BioHEL jobs across all CV folds and random seeds:

```bash
python run_biohel_hpc.py \
  --d <cv_data_directory> \
  --w <output_workspace> \
  --o <output_name> \
  --ol <outcome_label> \
  --il <instance_id_label> \
  --el <excluded_label> \
  --rc LSF \
  --rm 4 \
  --q i2c2_normal \
  --cv 10 \
  --r 20 \
  --enable_rpe
```

**Parameters**:
- `--d`: CV data directory or single training file
- `--w`: Output workspace root
- `--o`: Output subdirectory name
- `--ol`: Column name of outcome/class label
- `--il`: Column name of instance IDs
- `--el`: Excluded column name (e.g., experimental group identifier)
- `--rc`: Resource controller (LSF for HPC)
- `--rm`: Memory request (GB)
- `--q`: HPC queue name
- `--cv`: Number of CV folds
- `--r`: Number of random seeds/replicates
- `--enable_rpe`: Enable BioHEL RPE post-processing

### Step 3: Run RIPPER Training

#### Option A: Single Debug Run (Local, No HPC)

For testing on a single fold without HPC submission:

```bash
python job_ripper_hpc.py \
  --d <training_data_file> \
  --o <output_directory> \
  --ol <outcome_label> \
  --il <instance_id_label> \
  --el <excluded_label> \
  --rs <random_seed> \
  --verbosity 0 \
  --v
```

#### Option B: Full Batch Submission (HPC/LSF)

Submit RIPPER jobs across all CV folds and random seeds:

```bash
python run_ripper_hpc.py \
  --d <cv_data_directory> \
  --w <output_workspace> \
  --o <output_name> \
  --ol <outcome_label> \
  --il <instance_id_label> \
  --el <excluded_label> \
  --rc LSF \
  --rm 4 \
  --q i2c2_normal \
  --cv 10 \
  --r 20 \
  --verbosity 0
```

**Parameters**: Same as BioHEL except:
- `--verbosity`: Output verbosity level (0 = minimal, 1+ = verbose)
- No `--enable_rpe` option (RIPPER doesn't use post-processing)

### Step 4: Aggregate Results

Once individual training jobs complete, aggregate results across all folds and seeds.

#### Aggregate BioHEL Results

```bash
python run_biohel_sum_hpc.py \
  --w <output_workspace> \
  --o <output_name> \
  --rc LSF \
  --rm 4 \
  --q i2c2_normal \
  --cv 10 \
  --r 20
```

#### Aggregate RIPPER Results

```bash
python run_ripper_sum_hpc.py \
  --w <output_workspace> \
  --o <output_name> \
  --rc LSF \
  --rm 4 \
  --q i2c2_normal \
  --cv 10 \
  --r 20 \
  --plots
```

### Step 5: Generate Paper Results Tables

Generate comparative performance tables comparing all algorithms:

```bash
python job_sum_table_hpc.py \
  --biohel_root <biohel_output_workspace> \
  --ripper_root <ripper_output_workspace> \
  --out <output_directory>
```

### Step 6: Statistical Analysis (Optional)

Compare HEROS baseline against BioHEL and RIPPER using Wilcoxon signed-rank tests:

```bash
# Combine HEROS results
python stats_scripts/combine_heros_for_stats.py \
  --root <heros_output_directory> \
  --outdir <output_directory> \
  --outname combined_heros_cv_default_runs_long.csv

# Run statistical tests
python stats_scripts/job_statistical_test.py \
  --heros_csv <heros_results_csv> \
  --other_csv <biohel_ripper_results_csv> \
  --outdir <output_directory> \
  --prefix heros_baseline_wilcoxon \
  --alpha 0.05
```

## Reproducing Paper Results

### Quick Reference: Complete Experiment Workflow

```bash
# 1. Create CV folds (prerequisite, shared across all algorithms)
python ../cv_partitioning/run_CV_Partitioner.py ...

# 2. Submit BioHEL training jobs
python run_biohel_hpc.py ...

# 3. Submit RIPPER training jobs
python run_ripper_hpc.py ...

# 4. Wait for all jobs to complete
python check_failed_jobs.py

# 5. Aggregate results
python run_biohel_sum_hpc.py ...
python run_ripper_sum_hpc.py ...

# 6. Generate paper tables
python job_sum_table_hpc.py ...

# 7. Optional: Statistical significance testing
python stats_scripts/job_statistical_test.py ...

# 8. Optional: Cleanup scratch files
python cleanup_hpc_artifacts.py --w <output_workspace> --yes
```

### All Experiment Commands

**All commands needed to reproduce the exact paper results are documented in `run_commands.txt`.**

This file contains the complete, tested pipeline including:
- CV fold creation commands with exact dataset paths
- BioHEL debug runs and full training pipelines (multiplexer and GAMETES datasets)
- RIPPER debug runs and full training pipelines
- Result aggregation steps
- Paper table generation
- Statistical significance testing
- Optional cleanup commands

**Always reference `run_commands.txt` for the authoritative commands used in the paper.**

## Monitoring and Troubleshooting

### Check Job Status

```bash
# List all current jobs
bjobs

# Check specific queue
bqueues

# List failed jobs
python check_failed_jobs.py
```

### BioHEL Binary Not Found

```bash
# Verify BioHEL was built successfully
./biohel --help

# Check if binary is in PATH
which biohel

# If not found, build from source and create symlink or copy to this directory
ln -s /path/to/built/biohel ./biohel
```

### RIPPER/Wittgenstein Import Errors

```bash
# Reinstall or upgrade Wittgenstein
pip install --upgrade wittgenstein

# Verify installation
python -c "import wittgenstein; print('OK')"
```

### BioHEL RPE Errors (Optional)

If using `--enable_rpe`:

```bash
# Verify postprocess binary exists and is executable
./postprocess --help

# Check postprocess.conf for correct configuration
cat postprocess.conf
```

### HPC Job Failures

- Check job logs in the output directory
- Verify memory and time requirements match queue limits
- Ensure dataset labels (`--ol`, `--il`, `--el`) are consistent across runs
- Use `check_failed_jobs.py` to identify which folds/seeds failed
- Check shared storage accessibility and permissions

### CV Fold Mismatch

- Ensure CV folds are created with matching parameters across algorithms
- Verify dataset labels are consistent between CV partitioning and training
- Check file paths in `run_commands.txt` match your environment

Statistical analysis uses Wilcoxon signed-rank tests for paired comparisons.

## References

- **BioHEL**: https://ico2s.org/software/biohel.html
- **BioHEL RPE**: http://ico2s.org/software/biohel-rpe.html
- **Wittgenstein (RIPPER Python package)**: https://github.com/imoscovitz/wittgenstein
- **RIPPER Original Paper**: Cohen, W.W. (1995). Fast Effective Rule Induction. In *Proceedings of the 12th International Conference on Machine Learning (ICML)*

For HEROS training, see parent `../` directory for HEROS-specific runners.

## Contact & Support

For issues with:
- **BioHEL**: See https://ico2s.org/software/biohel.html
- **BioHEL RPE**: See http://ico2s.org/software/biohel-rpe.html
- **RIPPER/Wittgenstein**: See https://github.com/imoscovitz/wittgenstein
- **This pipeline**: Check `run_commands.txt` for exact commands and refer to script documentation

---
