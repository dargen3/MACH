#!/bin/bash

PARA=$(basename "$1")
SDF=$(basename "$2")
CHG=$(basename "$3")

PARA_name=${PARA:4:-4}
date=$(date +%Y_%m_%d_%H_%M_%S_$PARA_name)"_par"
ssh dargen3@nympha.metacentrum.cz "cd /storage/praha1/home/dargen3/mach ; mkdir $date"




printf "Copying of data to MetaCentrum...\n"
scp "$1" "$2" "$3" dargen3@nympha.metacentrum.cz:/storage/praha1/home/dargen3/mach/$date
printf "\e[32mok\e[39m\n\n"
printf "Connection to MetaCentrum...\n"
ssh dargen3@nympha.metacentrum.cz "cd /storage/praha1/home/dargen3/mach/para_submit; ./para_submit.sh $PARA $SDF $CHG $date $5 \"${4}\" -l select=1:ncpus=$5:mem=$6gb:scratch_local=5gb -l walltime=$7:00:00" && printf "Job is in planning system.\n\n"


