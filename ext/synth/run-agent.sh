#!/bin/bash
# The memo REPL agent on fenrir: run-agent.sh NAME MODEL [extra -J options...].
# CPU llama.cpp v0.7.0 (the GPU needs a reset), niced, under guard.sh. The
# tool host is a second JVM the driver starts (memo.synth.tools).
source ~/mlx/gen-clj-tools/env.sh
cd ~/mlx/memo.clj-hint/ext/synth
L=~/mlx/gen-clj-logs/memo-synth
NAME=$1; MODEL=$2; shift 2
echo "== $NAME $(date) load: $(cut -d' ' -f1-3 /proc/loadavg)" > $L/$NAME.txt
GUARD_SECONDS=${GUARD_SECONDS:-21600} GPU_LIMIT_GB=1 MIN_AVAIL_GB=24 ~/mlx/gen-clj-tools/guard.sh 56 $L/$NAME.guard \
  nice -n 10 clojure -J-Dgen.llama.lib=$HOME/mlx/gen.clj-m3/native/build/llama/linux-x86_64-cpu/lib \
  -J-Dsynth.model=$MODEL -J-Dsynth.out=$L/$NAME "$@" -M:driver -m memo.synth.agent >> $L/$NAME.txt 2>&1
echo "== $NAME exit $? $(date)" >> $L/$NAME.txt
