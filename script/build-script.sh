#!/usr/bin/env bash
set -x
# Jenkins style
# CHANGED_MODEL_FILES=`git diff --stat --name-only $GIT_PREVIOUS_COMMIT $GIT_COMMIT | grep model | grep py | grep -v topoml_util`

echo "Changes:"
cat $1

# TeamCity style
CHANGED_MODEL_FILES="$(cat $1 | cut -d \: -f 1 | grep model | grep py | grep -v topoml_util)"
echo ${CHANGED_MODEL_FILES}

set -e
cd model
for FILE in ${CHANGED_MODEL_FILES}
do
	python3 ../${FILE}
done

echo "built!"