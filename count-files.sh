#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <host> <remote_directory>"
    exit 1
fi

SSH_SERVER="$1"
REMOTE_DIR="$2"


ssh "$SSH_SERVER" "
find '$REMOTE_DIR' -type d | while read DIR; do
    COUNT=\$(find \"\$DIR\" -maxdepth 2 -type f -iname '*.jpg' | wc -l)
    echo \"\$DIR: \$COUNT\"
done
"