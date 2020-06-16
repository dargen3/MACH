python mach.py --mode comparison --emp_chgs_file results/alexandra/small_set/parameterizations/peptides_plain-ba/results_data/npa_out_SQE.chg  --ref_chgs_file  results/alexandra/small_set/parameterizations/peptides_plain-ba/npa_out.chg  --data_dir smazat_dir  --sdf_file results/alexandra/small_set/parameterizations/peptides_plain-ba/peptides.sdf  --ats_types_pattern plain-ba -f 

python mach.py --mode calculation --chg_method SQE --sdf_file results/alexandra/small_set/peptides.sdf  --params_file results/alexandra/proteins/usable/parameterizations/24_plain_ba_sb/results_data/parameters.json --data_dir smazat_dir -f 

python mach.py --mode parameterization --chg_method SQE --optimization_method guided_minimization --sdf_file results/alexandra/small_set/peptides.sdf --ref_chgs_file results/alexandra/small_set/npa_out.chg --data_dir smazat_dir --params_file results/alexandra/small_set/parameterizations/peptides_plain-ba/results_data/parameters.json  --subset 20 -f --num_of_samples 50 --num_of_candidates 2

