#!/bin/bash

python3 save_prec_models.py -y 'amps' -f 'X_6d_theta' 
python3 save_prec_models.py -y 'abs_err' -f 'X_6d_theta' 
# python3 save_prec_models.py -y 't_emop' -f 'X_6d_theta' 

python3 save_prec_models.py -y 'amps' -f 'X_7d_ISCO' 
python3 save_prec_models.py -y 'abs_err' -f 'X_7d_ISCO' 
# python3 save_prec_models.py -y 't_emop' -f 'X_7d_ISCO' 

# -f 'X_7d_ISCO' 'X_6d_theta'
# -y 'amps' 't_emop' 'abs_err'