#!/bin/bash 
# This script will first sync the results/data folder with the driectory in 
# variable . Then it will then push the results to github.
DESTINATION_DIR="/home/priyam/projects/marl_ccm"
SOURCE_DIR="/home/priyam/projects/predator-prey/results/data"

rsync -av $SOURCE_DIR $DESTINATION_DIR
cd $DESTINATION_DIR
git add -A
git commit -m "syncing results"
git push origin main