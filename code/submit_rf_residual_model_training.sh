#!/bin/bash
# submit_rf_residual_model_training.sh
# - Submits an array job over locations.txt
# - Writes ALL out/err logs to an ABSOLUTE directory on Turbo (permission-safe)
# - Runs the job from your code directory

set -e
set -o pipefail

# -------- Absolute paths --------
CODE_DIR="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/code"   # <-- adjust if needed
LOCATIONS_FILE="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/locations.txt"
PY_SCRIPT="${CODE_DIR}/rf_residual_model_training.py"

# Logs must be created (permission-safe)
LOG_DIR="${CODE_DIR}/logs/training"
mkdir -p "${LOG_DIR}"

# -------- Slurm resources --------
JOB_NAME="rf_resid_train"
TIME_LIMIT="48:00:00"
MEMORY="30G"
CPUS=4
PARTITION="standard"
MAIL_TYPE="END,FAIL"

# -------- Checks --------
if [[ ! -f "${LOCATIONS_FILE}" ]]; then
  echo "ERROR: locations file not found: ${LOCATIONS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: python script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

num_lines=$(wc -l < "${LOCATIONS_FILE}" | tr -d ' ')
if [[ "${num_lines}" -le 0 ]]; then
  echo "ERROR: locations file is empty: ${LOCATIONS_FILE}" >&2
  exit 1
fi
max_index=$((num_lines - 1))

echo "Submitting array job with ${num_lines} tasks (index 0..${max_index})"
echo "CODE_DIR       : ${CODE_DIR}"
echo "Locations file : ${LOCATIONS_FILE}"
echo "Python script  : ${PY_SCRIPT}"
echo "Log dir        : ${LOG_DIR}"

# -------- Submit inline sbatch script --------
sbatch \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/rf_resid_%A_%a.out" \
  --error="${LOG_DIR}/rf_resid_%A_%a.err" \
  --time="${TIME_LIMIT}" \
  --mem="${MEMORY}" \
  --cpus-per-task="${CPUS}" \
  --partition="${PARTITION}" \
  --mail-type="${MAIL_TYPE}" \
  --array="0-${max_index}" \
  --chdir="${CODE_DIR}" \
  --export=ALL,LOCATIONS_FILE="${LOCATIONS_FILE}",PY_SCRIPT="${PY_SCRIPT}" <<'SBATCH_EOF'
#!/bin/bash

# IMPORTANT:
# Do NOT use "set -u" here. Some conda activate hooks reference unset vars.
set -e
set -o pipefail

module purge
module load python

# ---- Conda init for non-interactive Slurm shells (fixes: activate not found) ----
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /home/ziqiwe/.conda/envs/ziqiLE

# Defensive checks (without set -u)
if [[ -z "${LOCATIONS_FILE}" ]]; then
  echo "ERROR: LOCATIONS_FILE is not set" >&2
  exit 1
fi
if [[ -z "${PY_SCRIPT}" ]]; then
  echo "ERROR: PY_SCRIPT is not set" >&2
  exit 1
fi

location=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${LOCATIONS_FILE}" | tr -d '\r')
if [[ -z "${location}" ]]; then
  echo "ERROR: Empty location for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID} location: [${location}]"
echo "Host: $(hostname)"
echo "CWD : $(pwd)"
echo "Script: ${PY_SCRIPT}"
python -c "import sys; print('python exe:', sys.executable)"

python "${PY_SCRIPT}" --location "${location}"

SBATCH_EOF

