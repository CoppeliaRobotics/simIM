#!/bin/sh

if [ "$(uname)" = "Darwin" ]; then
    VREP_BIN=$VREP_ROOT/vrep.app/Contents/MacOS/vrep
else
    VREP_BIN=$VREP_ROOT/vrep.sh
fi

TEST_DIR="$(cd "$(dirname "$0")"; pwd)"

$VREP_BIN -h -s1000 -q -g$TEST_DIR/$1 $TEST_DIR/runtest.ttt
