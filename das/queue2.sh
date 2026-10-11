#!/bin/bash
# usage: queue2.sh <jobfile> <parallel>   each line: a full command run from das/ (e.g. "MODULE=das.bench_mbpp ./run.sh ...")
cd /root/aditya/das
N=${2:-2}
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
  eval "$line" &
  sleep 20
  while [ "$(jobs -r | wc -l)" -ge "$N" ]; do sleep 10; done
done < "$1"
wait
