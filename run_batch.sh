#!/bin/bash

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
            N_BLOCKS="$2"
            shift 2;;
        -n | --n_blocks) 
            STIME="$2"
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


CMD="python src/model/main.py"
DATADIR="store_$(date +%H%M%S)"

# clean
rm -fr log_* >/dev/null 2>&1

# run first block
$CMD -t $STIME -d
CURR_TIME=$(date +%H%M%S)
for f in log_*; do
    mv $f $DATADIR/$f_$CURR_TIME 
done
cp dumped_robot  $DATADIR/$dumped_robot_$CURR_TIME 

# run n blocks
if [ $N_BLOCKS -gt 1 ]; then
    for((n=0;n<$[N_BLOCKS-1];n++)); do
        # run n-th block
        $CMD -t $STIME -d -l 
        CURR_TIME=$(date +%H%M%S)
        for f in log_*; do
            cp $f $DATADIR/$f_$CURR_TIME 
        done
        cp dumped_robot  $DATADIR/$dumped_robot_$CURR_TIME 
    done
fi



