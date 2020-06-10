#!/bin/bash

COMMAND=$1
DIR=$2
qsub_args="${@:3}"

data_dir=/storage/praha1/home/dargen3/mach/$DIR
script_file=$(mktemp)


cat << EOF > $script_file
#!/bin/bash
trap "clean_scratch" TERM EXIT
cp -r $data_dir/input_files/* $data_dir/../mach.py $data_dir/../modules* \$SCRATCHDIR || exit 1
cd \$SCRATCHDIR || exit 2

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COMMAND
cd $DIR/output_files
cp * $data_dir/output_files || export CLEAN_SCRATCH=false
EOF
qsub -N parameterization  $qsub_args $script_file
rm $script_file

