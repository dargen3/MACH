#!/bin/bash

PARA=$1
PARA_name=${PARA:4:-4}
SDF=$2
CHG=$3
COMMAND=$5
DIR=$4
qsub_args="${@:6}"

data_dir=/storage/praha1/home/dargen3/mach/$DIR
script_file=$(mktemp)


cat << EOF > $script_file
#!/bin/bash
trap "clean_scratch" TERM EXIT
cp -r $data_dir/$PARA $data_dir/$SDF $data_dir/../mach.py $data_dir/../modules* \$SCRATCHDIR || exit 1
cd \$SCRATCHDIR || exit 2

module add anaconda3-4.0.0
source /storage/praha1/home/dargen3/.conda/envs/dargen3_conda/bin/activate /storage/praha1/home/dargen3/.conda/envs/dargen3_conda
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COMMAND
cp *.chg $data_dir || export CLEAN_SCRATCH=false
EOF

qsub -N $PARA_name $qsub_args $script_file
rm $script_file

