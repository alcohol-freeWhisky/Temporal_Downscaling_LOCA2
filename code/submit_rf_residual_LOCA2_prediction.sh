#!/bin/bash
# submit_rf_residual_LOCA2_prediction.sh
#
# - Submits an array job over counties.txt (one county per task)
# - Uses enforced artifacts by default (can change ARTIFACT_SUBDIR)
# - No array concurrency cap (removed "%10" limit)
# - Runs from your code directory; logs to an absolute, permission-safe path

set -e
set -o pipefail

# -------- Absolute paths --------
CODE_DIR="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/code"   # <-- adjust if needed
COUNTIES_FILE="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/county_list_rf_model_check.txt"  # one county id per line (e.g., G0100730)
PY_SCRIPT="${CODE_DIR}/rf_residual_LOCA2_prediction.py"

INPUT_ROOT="/nfs/turbo/seas-mtcraig-climate/Mai/LOCA2_point_csv_2018_2062"
MODEL_ROOT="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_model"
OUTPUT_ROOT="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_downscaling"

# Use enforced artifacts
ARTIFACT_SUBDIR="enforced"

# Logs directory
LOG_DIR="${CODE_DIR}/logs/loca2_prediction"
mkdir -p "${LOG_DIR}"

# -------- Slurm resources --------
JOB_NAME="rf_loca2_pred"
TIME_LIMIT="48:00:00"
MEMORY="30G"
CPUS=8
PARTITION="standard"
MAIL_TYPE="END,FAIL"

# -------- Checks --------
if [[ ! -f "${COUNTIES_FILE}" ]]; then
  echo "ERROR: counties file not found: ${COUNTIES_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: python script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

num_lines=$(wc -l < "${COUNTIES_FILE}" | tr -d ' ')
if [[ "${num_lines}" -le 0 ]]; then
  echo "ERROR: counties file is empty: ${COUNTIES_FILE}" >&2
  exit 1
fi
max_index=$((num_lines - 1))

echo "Submitting array job with ${num_lines} counties (index 0..${max_index})"
echo "CODE_DIR       : ${CODE_DIR}"
echo "COUNTIES_FILE  : ${COUNTIES_FILE}"
echo "PY_SCRIPT      : ${PY_SCRIPT}"
echo "INPUT_ROOT     : ${INPUT_ROOT}"
echo "MODEL_ROOT     : ${MODEL_ROOT}"
echo "OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "ARTIFACT_SUBDIR: ${ARTIFACT_SUBDIR}"
echo "LOG_DIR        : ${LOG_DIR}"

# -------- Submit inline sbatch script --------
sbatch \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/pred_%A_%a.out" \
  --error="${LOG_DIR}/pred_%A_%a.err" \
  --time="${TIME_LIMIT}" \
  --mem="${MEMORY}" \
  --cpus-per-task="${CPUS}" \
  --partition="${PARTITION}" \
  --mail-type="${MAIL_TYPE}" \
  --array="0-${max_index}" \
  --chdir="${CODE_DIR}" \
  --export=ALL,COUNTIES_FILE="${COUNTIES_FILE}",PY_SCRIPT="${PY_SCRIPT}",INPUT_ROOT="${INPUT_ROOT}",MODEL_ROOT="${MODEL_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",ARTIFACT_SUBDIR="${ARTIFACT_SUBDIR}" \
<<'SBATCH_EOF'
#!/bin/bash
set -e
set -o pipefail

module purge
module load python

# ---- Conda init for non-interactive Slurm shells ----
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /home/ziqiwe/.conda/envs/ziqiLE

# Get county for this array task
county=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${COUNTIES_FILE}" | tr -d '\r')
if [[ -z "${county}" ]]; then
  echo "ERROR: Empty county for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID} county: ${county}"
echo "Host: $(hostname)"
echo "CWD : $(pwd)"
echo "python exe: $(python -c 'import sys; print(sys.executable)')"

python -u "${PY_SCRIPT}" \
  --county "${county}" \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_root "${MODEL_ROOT}" \
  --artifact_subdir "${ARTIFACT_SUBDIR}"

SBATCH_EOF

