DATA160=dir_to/ILSVRC/Data/CLS-LOC
DATA352=dir_to/ILSVRC/Data/CLS-LOC
DATA=dir_to/ILSVRC/Data/CLS-LOC

NAME=2px
LAMBDA=20000
DECAYRATE=0.99

CONFIG1=configs/configs_ELLE-A_${NAME}_phase1_scar.yml
CONFIG2=configs/configs_ELLE-A_${NAME}_phase2_scar.yml
CONFIG3=configs/configs_ELLE-A_${NAME}_phase3_scar.yml

PREFIX1=ELLE-A_adv_phase1_${NAME}_lambda${LAMBDA}_rep2
PREFIX2=ELLE-A_adv_phase2_${NAME}_lambda${LAMBDA}_rep2
PREFIX3=ELLE-A_adv_phase3_${NAME}_lambda${LAMBDA}_rep2

OUT1=ELLE-A_train_phase1_${NAME}_lambda${LAMBDA}_rep2.out
OUT2=ELLE-A_train_phase2_${NAME}_lambda${LAMBDA}_rep2.out
OUT3=ELLE-A_train_phase3_${NAME}_lambda${LAMBDA}_rep2.out

EVAL1=ELLE-A_eval_phase1_${NAME}_lambda${LAMBDA}_rep2.out
EVAL2=ELLE-A_eval_phase2_${NAME}_lambda${LAMBDA}_rep2.out
EVAL3=ELLE-A_eval_phase3_${NAME}_lambda${LAMBDA}_rep2.out

END1=dir_to/ImageNet/trained_models/ELLE-A_adv_phase1_${NAME}_lambda${LAMBDA}_rep2_step2_eps2_repeat1/checkpoint_epoch6.pth.tar
END2=dir_to/ImageNet/trained_models/ELLE-A_adv_phase2_${NAME}_lambda${LAMBDA}_rep2_step2_eps2_repeat1/checkpoint_epoch12.pth.tar
END3=dir_to/ImageNet/trained_models/ELLE-A_adv_phase3_${NAME}_lambda${LAMBDA}_rep2_step2_eps2_repeat1/checkpoint_epoch15.pth.tar

# training for phase 1
python3 -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --elle_lambda $LAMBDA --decay_rate $DECAYRATE | tee $OUT1

# evaluation for phase 1
# python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
python3 -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 --elle_lambda $LAMBDA --decay_rate $DECAYRATE | tee $OUT2

# evaluation for phase 2
# python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
python3 -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 --elle_lambda $LAMBDA --decay_rate $DECAYRATE | tee $OUT3

# evaluation for phase 3
# python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3
