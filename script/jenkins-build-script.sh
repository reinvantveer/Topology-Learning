#!/usr/bin/env bash
set -e
CHANGED_MODEL_FILES=`git diff --stat --name-only $GIT_PREVIOUS_COMMIT $GIT_COMMIT | grep model | grep -v topoml_util`
cd model
for FILE in ${CHANGED_MODEL_FILES}
do
	python3 ../${FILE}
done

echo "built!"