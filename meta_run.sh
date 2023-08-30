#!/bin/bash


# Argument 1: Package name, no default value so it's mandatory
PACKAGE_NAME=${1}

# Argument 2: Sweep ID, no default value so it's mandatory
SWEEP_ID=${2}

# Argument 3: The sweep configuration file
CONFIG_FILE=${3}

# Argument 4: Maximum number of jobs to execute, default is 1000
MAX_JOBS=${4:-1000}

# Argument 5: Device index, default is 0
DEVICE_INDEX=${5:-0}

# Check if SWEEP_ID and PACKAGE_NAME are provided
if [[ -z "$SWEEP_ID" ]] || [[ -z "$PACKAGE_NAME" ]]; then
  echo "Error: Sweep ID and Package Name are required."
  exit 1
fi

# Running the specified command with the given and default arguments
dysweep_run_resume --package $PACKAGE_NAME \
                   --function dysweep_compatible_run \
                   --run_additional_args gpu_index:$DEVICE_INDEX \
                   --config $CONFIG_FILE \
                   --sweep_id $SWEEP_ID \
                   --count $MAX_JOBS
