#!/usr/bin/env bash
set -ex
# Jenkins style
# CHANGED_MODEL_FILES=`git diff --stat --name-only $GIT_PREVIOUS_COMMIT $GIT_COMMIT | grep model | grep py | grep -v topoml_util`

# TeamCity style
CHANGED_MODEL_FILES="$(cat $1 | cut -d \: -f 1 | grep model | grep py | grep -v topoml_util)"

echo ${CHANGED_MODEL_FILES}

cd model
for FILE in ${CHANGED_MODEL_FILES}
do
	python3 ../${FILE}
done

echo "built!"