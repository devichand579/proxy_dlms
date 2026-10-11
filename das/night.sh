#!/bin/bash
cd /root/aditya/das
./queue2.sh jobs_night.txt 2
./queue2.sh jobs_timing.txt 1
echo "NIGHT DONE $(date)"
