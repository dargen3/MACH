#!/bin/bash
# only for my usage
PARA=$(basename "$1")
SDF=$(basename "$2")
CHG=$(basename "$3")
PARA_name=${PARA:4:-4}
date=$(date +%Y_%m_%d_%H_%M_%S_$PARA_name)"_cal"


ssh dargen3@tarkil.grid.cesnet.cz "cd mach ; mkdir $date"


printf "Copying of data to MetaCentrum...\n"
scp "$1" "$2" dargen3@tarkil.grid.cesnet.cz:/storage/praha1/home/dargen3/mach/$date
printf "\e[32mok\e[39m\n\n"
printf "Connection to MetaCentrum...\n"
ssh dargen3@tarkil.grid.cesnet.cz "cd mach/para_submit; ./para_submit_calc.sh $PARA $SDF $CHG $date \"${4}\" -l select=1:ncpus=1:mem=$5gb:scratch_local=1gb -l walltime=$6:00:00"
printf "Job is in planning system.\n\n"


