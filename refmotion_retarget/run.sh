#!/usr/bin/env bash
#config: --utf-8
##!/bin/pyenv python

set -e 

robot_id="1"
motion_name=""
fit=""
gmr=""
shape=""
vis=""
save_only=""
python_bin="python"
while getopts "r:m:fsgvo" arg
do 
case $arg in 
r)
robot_id=$OPTARG
;;
m)
motion_name=$OPTARG
;;
f)
fit=1
;;
g)
gmr=1
;;
s)
shape=1
;;
v)
vis=1
;;
o)
save_only=1
;;
?)
echo "Unknow args"
;;
esac
done

echo "Robot id is ${robot_id} and Motion name is $motion_name in the retarget."


if [ -n "$shape" ]; then

    echo "fit smpl shape ..."
python -u fit_smpl_shape.py robot=${robot_id}_fitting
fi


if [ -n "$fit" ]; then

    echo "fit smpl motion ..."
python -u fit_smpl_motion.py robot=${robot_id}_fitting +motion_name=$motion_name
fi

if [ -n "$gmr" ]; then

    echo "fit smplx motion using gmr..."
python -u fit_smplx_motion.py robot=${robot_id}_fitting +motion_name=$motion_name
fi


if [ -n "$vis" ]; then
    echo "convert and vis retarget motion in mujoco ..."
    if [ "$(uname -s)" = "Darwin" ] && command -v mjpython >/dev/null 2>&1; then
        python_bin="mjpython"
    fi
    "$python_bin" -u ./convert_in_mj.py robot=${robot_id}_fitting +motion_name=$motion_name
fi

if [ -n "$save_only" ]; then
    echo "convert retarget motion and save pkl ..."
    python -u ./convert_in_mj.py robot=${robot_id}_fitting +motion_name=$motion_name +save_only=true
fi
