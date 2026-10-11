#!/bin/bash
# Wait for the running queues, then re-run every job (finished batches are skipped; crashed runs resume).
cd /root/aditya/das
while pgrep -f "queue.sh jobs_(curve|pace3|gsm2).txt" > /dev/null; do sleep 60; done
./queue.sh jobs_curve.txt 3
./queue.sh jobs_pace3.txt 3
MODULE=das.bench_gsm8k ./queue.sh jobs_gsm2.txt 3
