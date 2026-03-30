#!/bin/bash
cd /home/hpc4090/asif_tanzila/TSNN-Drug

echo "Waiting for V1 pipeline to finish..."
while pgrep -f "run_full_pipeline.sh" > /dev/null 2>&1; do
    sleep 30
done

echo "V1 done. Starting V2 at $(date)"
bash scripts/run_v2_improved.sh
