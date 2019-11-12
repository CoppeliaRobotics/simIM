#!/bin/sh

if [ "$(uname)" = "Darwin" ]; then
    COPPELIASIM_BIN=$COPPELIASIM_ROOT_DIR/coppeliaSim.app/Contents/MacOS/coppeliaSim
else
    COPPELIASIM_BIN=$COPPELIASIM_ROOT_DIR/coppeliaSim.sh
fi

TEST_DIR="$(cd "$(dirname "$0")"; pwd)"

$COPPELIASIM_BIN -h -s1000 -q -g$TEST_DIR/$1 $TEST_DIR/runtest.ttt
