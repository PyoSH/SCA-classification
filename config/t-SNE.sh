#!/bin/bash

echo "Starting all t-SNE experiments..."

PYTHON_EXEC="/opt/anaconda3/envs/audio_DL/bin/python"
SCRIPT_PATH="/Users/seunghyunpyo/PycharmProjects/rnn_followup/t-SNE.py"
CONFIG_DIR="/Users/seunghyunpyo/PycharmProjects/rnn_followup/config"

# --- 각 실험 실행 ---
echo "Running experiment: B1 (mfcc)"
$PYTHON_EXEC /Users/seunghyunpyo/PycharmProjects/rnn_followup/t-SNE_mfcc.py --cfg $CONFIG_DIR/exp-B1.yaml

echo "Running experiment: B2-small"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-B2-small.yaml

echo "Running experiment: B2-middle"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-B2-middle.yaml

echo "Running experiment: B2-large"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-B2-large.yaml

echo "Running experiment: B3"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-B3.yaml

echo "Running experiment: B4"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-B4.yaml

echo "Running experiment: P"
$PYTHON_EXEC $SCRIPT_PATH --cfg $CONFIG_DIR/exp-P.yaml

# --- 모든 실험 종료 ---
echo "All experiments finished!"