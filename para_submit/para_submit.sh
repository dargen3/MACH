#!/bin/bash

PARA=$1
SDF=$2
PARA_name=${PARA::3}"_"$SDF
CHG=$3
COMMAND=$6
DIR=$4
qsub_args="${@:7}"

data_dir=/storage/praha1/home/dargen3/mach/$DIR
script_file=$(mktemp)


cat << EOF > $script_file
#!/bin/bash
trap "clean_scratch" TERM EXIT
cp -r $data_dir/$PARA $data_dir/$SDF $data_dir/$CHG $data_dir/../mach.py $data_dir/../modules* \$SCRATCHDIR || exit 1
cd \$SCRATCHDIR || exit 2

#module add anaconda3-4.0.0
#source /storage/praha1/home/dargen3/.conda/envs/dargen3_conda/bin/activate /storage/praha1/home/dargen3/.conda/envs/dargen3_conda

module add conda-modules-py37
source /storage/praha1/home/dargen3/.conda/envs/dargen3_conda/bin/activate /storage/praha1/home/dargen3/.conda/envs/my_env_37


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COMMAND
cp -r results_data $data_dir || export CLEAN_SCRATCH=false
EOF
qsub -N $PARA_name $qsub_args $script_file
rm $script_file

