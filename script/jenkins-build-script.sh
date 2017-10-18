#!/usr/bin/env bash
CHANGED_FILES=`git diff --stat --name-only $GIT_PREVIOUS_COMMIT $GIT_COMMIT`
for FILE in ${CHANGED_FILES}
do
	echo ${FILE}
done

echo "built!"