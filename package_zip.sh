#!/bin/bash

current_datetime=$(date +'%Y-%m-%d_%H-%M-%S')
archive_name_prefix="archive"
zip -r "./dist/constrained-attacks_${archive_name_prefix}_${current_datetime}.zip" ./ -x "data/*" -x "bdd/*" ".git/*" -x "dist/*" -x "*/.mypy_cache/*" -x ".mypy_cache/*" -x "tmp/*" -x ".pytest_cache/*" -x ".cometml-runs/*" -x ".vscode/*" -x "*__pycache__*" -x ".idea/*" -x "doc/*"
