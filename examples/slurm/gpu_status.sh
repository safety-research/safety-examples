#!/bin/bash

printf "%-15s %10s %10s %10s\n" "NODE" "USED" "TOTAL" "FREE"
printf "%s\n" "------------------------------------------------"

for node in $(sinfo -h -o "%n" | sort -u); do
    info=$(scontrol show node "$node" 2>/dev/null)

    total=$(echo "$info" | grep "Gres=gpu:" | grep -oP 'gpu:\K[0-9]+')
    used=$(echo "$info" | grep "AllocTRES=" | grep -oP 'gres/gpu=\K[0-9]+')

    if [ -n "$total" ]; then
        used=${used:-0}
        free=$((total - used))
        printf "%-15s %10s %10s %10s\n" "$node" "$used" "$total" "$free"
    fi
done
