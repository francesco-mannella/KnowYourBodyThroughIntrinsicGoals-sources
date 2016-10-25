#!/bin/bash

set -e

usage()
{
cat << EOF

usage: $0 options

This script runs the robot simulation in batch mode and collects data 

OPTIONS:
   -t --stime       number of timesteps of a single simulation block
   -d --dumped      path/to/dumped_robot
   -h --help        show this help

EOF
}

STIME=100000
DUMPED=

# getopt
GOTEMP="$(getopt -o "t:d:h" -l "stime:,dumped:,help"  -n '' -- "$@")"

if ! [ "$(echo -n $GOTEMP |sed -e"s/\-\-.*$//")" ]; then
    usage; exit;
fi

eval set -- "$GOTEMP"


while true ;
do
    case "$1" in
       -t | --stime): 
            STIME="$2"
            shift 2;;
       -d | --dumped) 
            DUMPED="$2"
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


if ! [ -e $DUMPED ]; then
    DUMPED=
fi

if [ -z $DUMPED ]; then

    echo
    echo
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " You must choose a valid  dumped_robot file"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo
    usage; exit;
fi


CURRDIR=$(pwd)
TMPDIR="/tmp/robot_$(date +%H%M%S)"
mkdir $TMPDIR
ln -s $(pwd)/src $TMPDIR/src
cp $DUMPED $TMPDIR/dumped_robot
cd $TMPDIR

CMD="python src/model/main.py"

# clean
rm -fr log_* >/dev/null 2>&1

# run first block
$CMD -t $STIME -g -l 

cd $CURRDIR
rm -fr $TMPDIR

