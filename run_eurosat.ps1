# CoPrompt EuroSAT Runner for Windows
# Usage: .\run_eurosat.ps1

param (
    [int]$seed = 1,
    [string]$exp_name = "CoPrompt_Result"
)

# Configuration
$PYTHON = "C:\Users\Admin\AppData\Local\Programs\Python\Python38\python.exe"
$DATA = "../data/"
$TRAINER = "CoPrompt"
$DATASET = "eurosat"
$CFG = "coprompt"
$SHOTS = 16
$EXP_NAME = "CoPrompt_Result"

# Setup Environment
$env:PYTHONPATH += ";$PWD"
Write-Host "Setting PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Cyan

$seeds = @(11, 12, 13)

foreach ($seed in $seeds) {
    Write-Host "`n==========================================" -ForegroundColor Magenta
    Write-Host "Starting Experiment for Seed: $seed ($DATASET)" -ForegroundColor Magenta
    Write-Host "==========================================`n" -ForegroundColor Magenta

    # --- Phase 1: Training on Base Classes ---
    $TRAIN_DIR = "output/${EXP_NAME}/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}"
    Write-Host "--- [PHASE 1] Training on Base Classes ---" -ForegroundColor Yellow
    
    & $PYTHON train.py `
        --root $DATA `
        --seed $seed `
        --trainer $TRAINER `
        --dataset-config-file "configs/datasets/${DATASET}.yaml" `
        --config-file "configs/trainers/${CFG}.yaml" `
        --output-dir $TRAIN_DIR `
        DATALOADER.NUM_WORKERS 0 `
        DATASET.NUM_SHOTS $SHOTS `
        DATASET.SUBSAMPLE_CLASSES base

    # --- Phase 2: Evaluation on New (Novel) Classes ---
    $TEST_DIR = "output/${EXP_NAME}/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}"
    Write-Host "`n--- [PHASE 2] Evaluation on New Classes ---" -ForegroundColor Yellow

    & $PYTHON train.py `
        --root $DATA `
        --seed $seed `
        --trainer $TRAINER `
        --dataset-config-file "configs/datasets/${DATASET}.yaml" `
        --config-file "configs/trainers/${CFG}.yaml" `
        --output-dir $TEST_DIR `
        --model-dir $TRAIN_DIR `
        --load-epoch 8 `
        --eval-only `
        DATALOADER.NUM_WORKERS 0 `
        DATASET.NUM_SHOTS $SHOTS `
        DATASET.SUBSAMPLE_CLASSES new
}

Write-Host "`n--- Finalizing: Aggregating Results ---" -ForegroundColor Green
& $PYTHON aggregate_results.py --exp-name $EXP_NAME --dataset $DATASET --seeds $seeds
