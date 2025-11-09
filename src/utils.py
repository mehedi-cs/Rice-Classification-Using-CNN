import torch
from termcolor import colored
import seaborn as sns
import warnings

def setup_environment():
    """Setup environment and print hardware info."""
    warnings.filterwarnings('ignore')
    sns.set_style('darkgrid')

    print(colored('Your hardware uses : ', 'blue', attrs=['bold']))
    if torch.cuda.is_available():
        print(colored('GPU', 'green', attrs=['bold']))
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(colored('CPU', 'green', attrs=['bold']))
