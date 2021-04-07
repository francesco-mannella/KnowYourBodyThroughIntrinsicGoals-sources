#!/bin/bash
# 
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

