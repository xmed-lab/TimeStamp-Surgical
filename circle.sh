#!/bin/bash
export CUDA_VISILBE_DEVICES=1
for i in {1..5}
do
  cd feature_extract
  if [ $i -eq 1 ] ; then
    python uncertainty_semi.py --action train_pseudo --dataset cholec80 --arch casual --num_epochs 5
  else
    python uncertainty_semi.py --action train_pseudo --dataset cholec80 --arch casual --num_epochs 5 --pseudo
  fi
  python uncertainty_semi.py --action extract_pseudo --dataset cholec80 --arch casual &
  python uncertainty_semi.py --action extract_pseudo --dataset cholec80 --arch casual --target test_set &
  wait
  cd ..
  python -W ignore main.py --action train --extract pseudo --pseudo uncertainty --dataset cholec80 \
    --smooth --arch casual
done
