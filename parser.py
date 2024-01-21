import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from core.utils.train import SCHEDULERS

from core.utils.utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='ELLE(-A): Efficient local linearity regularization to overcome catastrophic overfitting')
    parser.add_argument('--resume', type=str2bool, default=False, help='If yes, specify the resume file name; no need to specify any other arguments')
    parser.add_argument('--resume_fname', type=str,default=None)
    parser.add_argument('--fname', type=str,default=None)

    
    # training 
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.') # 128
    parser.add_argument('--batch-size-validation', type=int, default=1024, help='Batch size for val and testing.') 
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer (SGD).') 
    parser.add_argument('--clip_grad', type=float, default=None, help='Gradient norm clipping.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of all epochs.') 
    
    parser.add_argument("--pretrained", type=str, help="pretrained model path; None if not using any pretrained model", default=None) 
    parser.add_argument("--fname_extra", type=str, help="Extra info the file fname", default='')
   
    
    parser.add_argument("--save_intermediate_models", type=int, default=0) # epoch interval for saving; if 0, not save 
    ### others
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.') # cosinew
    parser.add_argument('--nesterov', type=str2bool, default=False, help='Use Nesterov momentum.')
    
    parser.add_argument('--model', choices=MODELS, default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--beta', default=6.0, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.') # -1 then Madry's AT    
    parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.') 
    

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.') 
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    parser.add_argument('--data-dir', type=str, default='data/')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input before applying the model') 
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')   

    parser.add_argument('--input_noise_rate', type=str2float, default=0, help = 'factor multiplying attack_eps defining the size of the ball where we sample before the adversarial attack')

    parser.add_argument('--reg', type=str, default='None', help='one of None, LLR, CURE, GradAlign or ELLE') 

    parser.add_argument('--lin_reg', type=str2float, default=1, help='Regularizationn weight for overcomming catastrophic overfitting') 
    parser.add_argument('--n_triplets', type=int, default=1, help='how many triplets to choose for enforcing local linearity with ELLE') 


    parser.add_argument('--output', type=str, default='FAST_at_cosinew_scheduler_comparison', help='Output folder for experiments') 
    parser.add_argument('--clamp', type=str2bool, default=False, help='If true: project perturbation onto the ball') 
    parser.add_argument('--track_loss_alignment', type=str2bool, default=False, help='If true: track GradAlign regularization term') 
    parser.add_argument('--track_logits_alignment', type=str2bool, default=False, help='If true: track the gradient alignment of the output of a random class') 
    parser.add_argument('--track_lin_err', type=str2bool, default=False, help='If true: track the local-linear error of the loss and the logits') 
    parser.add_argument('--track_lin_err_05', type=str2bool, default=False, help='If true: track the local-linear error of the loss and the logits') 
    parser.add_argument('--track_lin_err_max_curve', type=str2bool, default=False, help='If true: track the local-linear error of the loss and the logits') 

    parser.add_argument('--SLAT', type=str2bool, default=False, help='Use SLAT?') 
    parser.add_argument('--BAT', type=str2bool, default=False, help='Use BAT?') 
    parser.add_argument('--GAT', type=str2bool, default=False, help='Use GAT?') 

    
    parser.add_argument('--lambda_schedule', type=str, default='onoff', help='constant (ELLE), onoff (ELLE-A)') 
    parser.add_argument('--decay_rate', type=str2float, default=0.99, help='decay rate after increasing lambda') 
    parser.add_argument('--sensitivity', type=str2float, default=2, help='factor multiplying the standard deviation') 
    parser.add_argument('--lambda_BAT', type=str2float, default=10, help='lambda parameter for BAT') 
    parser.add_argument('--attack_lr_BAT', type=str2float, default=5000/255, help='lambda parameter for BAT') 
    
    return parser
