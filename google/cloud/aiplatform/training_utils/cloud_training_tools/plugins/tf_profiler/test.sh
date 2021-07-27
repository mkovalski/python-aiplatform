#!/bin/bash

AIP_TENSORBOARD_RESOURCE_NAME=projects/388116905117/locations/us-central1/tensorboards/8468983914898128896 \
  AIP_EXPERIMENT_NAME='my-test-experiment' \
  AIP_TENSORBOARD_LOG_DIR='tmp/' \
  python request_sender.py
