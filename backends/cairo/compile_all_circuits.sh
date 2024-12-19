#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$BASE_DIR/circuits/projects"
COMPILED_DIR="$BASE_DIR/circuits/compiled_ops"

mkdir -p "$COMPILED_DIR"

for project in $(ls "$PROJECTS_DIR"); do
    project_path="$PROJECTS_DIR/$project"
    echo "Processing project: $project at $project_path"

    (cd "$project_path" && scarb build)

    if [ $? -eq 0 ]; then
        echo "Build successful for $project."
        
        json_files=("$project_path/target/dev/"*.sierra.json)
        if [ ${#json_files[@]} -gt 0 ]; then
            cp "${json_files[@]}" "$COMPILED_DIR/"
            echo "Files copied for $project."
        else
            echo "No .sierra.json files found to copy for $project."
        fi
    else
        echo "Build failed for $project."
    fi
done

echo "All projects processed."
