#!/bin/bash
# after the gap queue ends, rerun (one at a time) any job whose log has no RESULT; runs resume from saved batches
cd /root/aditya/das
until grep -q "GAPS DONE" /root/aditya/outputs/logs/gaps.log; do sleep 60; done
grep -v '^\s*#' jobs_gaps.txt | while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue
  n=$(echo "$line" | grep -o -- '--name [^ ]*' | cut -d' ' -f2)
  grep -q RESULT /root/aditya/outputs/logs/$n.log 2>/dev/null && continue
  echo "retry $n"; eval "$line"
done
echo RETRY DONE
