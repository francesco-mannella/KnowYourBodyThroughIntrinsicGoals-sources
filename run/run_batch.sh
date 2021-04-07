#!/bin/bash

# The MIT License (MIT)
# 
# Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com> 
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 

set -e

usage()
{
cat << EOF

usage: $0 options

This script runs the robot simulation in batch modeand collects data 

OPTIONS:
   -t --stime       number of timesteps of a single simulation block
   -n --n_blocks    number of simulation blocks
   -h --help        show this help

EOF
}

MAIN_DIR=..

STIME=100000
N_BLOCKS=1

# getopt
GOTEMP="$(getopt -o "t:n:h" -l "stime:,n_blocks,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"


while true ;
do
    case "$1" in
        -t | --stime) 
            STIME="$2"
            shift 2;;
         -n | --n_blocks) 
            N_BLOCKS="$2"
            shift 2;;
       -h | --help)
            echo "on help"
            usage; exit;
            shift;
            break;;
        --) shift ; 
            break ;;
    esac
done


#################################################################################

CMD="python $MAIN_DIR/src/model/main.py"
if ! [ -d $MAIN_DIR/stores  ]; then mkdir $MAIN_DIR/stores; fi
DATADIR="$MAIN_DIR/stores/store_$(date +%H%M%S)"
mkdir $DATADIR

# clean
rm -fr $MAIN_DIR/log_*

# run first block
$CMD -t $STIME -d

CURR_TIME=$(date +%H%M%S)
for f in $MAIN_DIR/log_*; do
    mv $f $DATADIR/$(basename $f)_${CURR_TIME} 
done
cp $MAIN_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 

# run n blocks
if [ $N_BLOCKS -gt 1 ]; then
    for((n=0;n<$[N_BLOCKS-1];n++)); do
        # run n-th block
        $CMD -t $STIME -d -l 
        CURR_TIME=$(date +%H%M%S)
        for f in $MAIN_DIR/log_*; do
            mv $f $DATADIR/$(basename $f)_${CURR_TIME} 
        done
        cp $MAIN_DIR/dumped_robot $DATADIR/dumped_robot_$CURR_TIME 
    done
fi



