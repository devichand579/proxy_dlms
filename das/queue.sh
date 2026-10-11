#!/bin/bash
# usage: queue.sh <jobfile> <parallel>   each line: bench args (shell-quoted)
cd /root/aditya/das
N=${2:-3}
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
  eval "./run.sh $line" &
  sleep 20
  while [ "$(jobs -r | wc -l)" -ge "$N" ]; do sleep 10; done
done < "$1"
wait
