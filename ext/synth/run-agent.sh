#!/bin/bash
# The memo REPL agent on a Linux machine: run-agent.sh NAME MODEL [extra -J options...].
# llama.cpp v0.7.0, LLAMA_BUILD=cpu (the default) or cuda, niced, under guard.sh.
# The tool host is a second JVM the driver starts (memo.synth.tools). Run from a
# deployed copy of memo.clj. Started again with the same NAME and seed, a run
# resumes (memo.synth.agent/done-episodes).
source ~/mlx/gen-clj-tools/env.sh
cd "$(dirname "$0")"
L=~/mlx/gen-clj-logs/memo-synth
NAME=$1; MODEL=$2; shift 2
BUILD=${LLAMA_BUILD:-cpu}
if [ "$BUILD" = cuda ]; then
  # llama.cpp's CUDA build links libcudart.so.13 and libcublas.so.13: the pip CUDA 13 libraries
  export LD_LIBRARY_PATH="$HOME/mlx/venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  GPU_GB=${GPU_LIMIT_GB:-48}
else
  GPU_GB=${GPU_LIMIT_GB:-1}
fi
echo "== $NAME $(date) $BUILD load: $(cut -d' ' -f1-3 /proc/loadavg)" >> $L/$NAME.txt
GUARD_SECONDS=${GUARD_SECONDS:-21600} GPU_LIMIT_GB=$GPU_GB MIN_AVAIL_GB=24 ~/mlx/gen-clj-tools/guard.sh 56 $L/$NAME.guard \
  nice -n 10 clojure -J-Dgen.llama.lib=$HOME/mlx/gen.clj-m3/native/build/llama/linux-x86_64-$BUILD/lib \
  -J-Dsynth.model=$MODEL -J-Dsynth.out=$L/$NAME "$@" -M:driver -m memo.synth.agent >> $L/$NAME.txt 2>&1
echo "== $NAME exit $? $(date)" >> $L/$NAME.txt
