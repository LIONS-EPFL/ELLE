DATA=../../../data_shared/imagenet/ILSVRC/Data/CLS-LOC

# 6eps evaluation
python3 main_fast.py $DATA --config configs/configs_ELLE_6px_evaluate.yml --output_prefix eval_6px_rep1 --resume trained_models/ELLE_adv_phase3_6px_lambda5000_scar_rep1_step6_eps6_repeat1/model_best.pth.tar --evaluate --restarts 10


# 4eps evaluation
python3 main_fast.py $DATA --config configs/configs_ELLE_4px_evaluate.yml --output_prefix eval_4px_rep1 --resume trained_models/ELLE_adv_phase3_4px_lambda5000_scar_rep1_step4_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10


# 2eps evaluation 
python3 main_fast.py $DATA --config configs/configs_ELLE_2px_evaluate.yml --output_prefix eval_2px_rep1 --resume trained_models/ELLE_adv_phase3_2px_lambda5000_scar_rep1_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10
