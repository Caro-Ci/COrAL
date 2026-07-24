import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
torch.cuda.empty_cache()
from dataset.multibench import MultiBenchDataModule
from pytorch_lightning import Trainer
from modules.coral import COrAL
from modules.mmfusion import MMFusion
from modules.transformer import Transformer
from modules.mlp import MLP
from modules.gru import GRU
from modules.eval_ortho import *
from modules.input_adapters import FeaturesInputAdapter
import warnings
from evaluation.linear_probe import evaluate_linear_probe
import argparse


def classification_scoring(model, data_module):
    Z_train, y_train = model.extract_features(data_module.train_dataloader())
    Z_val, y_val = model.extract_features(data_module.val_dataloader())
    Z_test, y_test = model.extract_features(data_module.test_dataloader())
    scores = evaluate_linear_probe(Z_train, y_train, Z_test, y_test, Z_val, y_val)
    return scores["acc1"]

def main(dataset):
    results = {'accuracies': [],
               'epoch_times': [],
               'acos_u1u2': [], 'grass_u1u2': [],
               'acos_su1': [], 'grass_su1': [],
               'acos_su2': [], 'grass_su2': []}

    for seed in range(41, 46):  
        print("seed:", seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed) 
        warnings.filterwarnings("ignore", category=UserWarning) 

        if dataset =="mosi":
            encoders=[Transformer(n_features=20, dim=40, max_seq_length=50, positional_encoding=False), 
                        Transformer(n_features=300, dim=40, max_seq_length=50, positional_encoding=False), ]
            input_adapters=[None, None]
            lin_layers= [nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20)),
                            nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20))]
            embed_dim= 40
            modalities = ["vision", "text"]
            augmentations = "drop+noise"
        elif dataset=="mosei":
            encoders=[Transformer(n_features=35, dim=40, max_seq_length=50, positional_encoding=False), 
                        Transformer(n_features=300, dim=40, max_seq_length=50, positional_encoding=False), ]
            input_adapters=[None, None]
            lin_layers= [nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20)),
                            nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20))]
            embed_dim= 40
            modalities = ["vision", "text"]
            augmentations = "drop+noise"
        elif dataset == "sarcasm" or dataset=="humor":
            encoders=[Transformer(n_features=371, dim=40, max_seq_length=50, positional_encoding=False), 
                        Transformer(n_features=300, dim=40, max_seq_length=50, positional_encoding=False),]
            input_adapters=[None, None]
            lin_layers= [nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20)),
                            nn.Sequential( nn.Linear(40, 40),nn.LayerNorm(40),nn.GELU(), nn.Linear(40, 20))]
            embed_dim= 40
            modalities = ["vision", "text"]
            augmentations = "drop+noise"
        elif dataset == "mimic":
            encoders=[MLP(indim=5, hiddim=10, outdim=10, dropout=False), 
                        GRU (indim=12, hiddim=512, dropout=False, batch_first=True),]
            input_adapters=[FeaturesInputAdapter(n_features=10,dim_tokens=512), None]
            lin_layers= [nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256)),
                            nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256))]
            embed_dim= 512
            modalities = ["tabular", "timeseries"]
            augmentations =["noise", "drop+noise"]


        data_module_training = MultiBenchDataModule(dataset, model="COrAL", 
                                                batch_size=64, num_workers=1, 
                                                modalities=modalities, 
                                                augmentations=augmentations)
        downstream_data = MultiBenchDataModule(dataset, model="Sup", 
                                                batch_size=64, num_workers=1, 
                                                modalities=modalities)
        
        coral= COrAL(
                encoder=MMFusion(
                    encoders=encoders,  
                    input_adapters=input_adapters,
                    lin_layers=lin_layers,
                    embed_dim=embed_dim
                ),
                projection=COrAL._build_mlp(embed_dim, 512, 256),
                uni_projection=COrAL._build_mlp(int(embed_dim/2), 512, 256),
                optim_kwargs=dict(lr=1e-4, weight_decay=1e-3),
                loss_kwargs=dict(temperature=0.1, ortho_weights=10, unique_weights=1,shared_weights=1),
                asym_masking = True)

        trainer = Trainer(inference_mode=False, max_epochs=100)
        trainer.fit(coral, datamodule=data_module_training)
        score_acc= classification_scoring(coral, downstream_data)
        print(f"COrAL accuracy on {dataset} lin probing={100 * score_acc:.1f}")
        results['accuracies'].append(score_acc)
        m_epoch_time= coral.median_epoch_time()
        results['epoch_times'].append(m_epoch_time)
       # if seed == 41:
       #    contour_train_test(coral,downstream_data, "density_plot_umap.svg",mod1_name="Vision", mod2_name="Text")
        y, shared, u1, u2 = coral.extract_all_the_features(downstream_data.test_dataloader())
        acos = cosine_u1_u2(u1, u2)
        results['acos_u1u2'].append(acos)
        results['grass_u1u2'].append(grassmann_distance(u1, u2, k=10))
        acos_su1 = cosine_u1_u2(shared, u1)
        acos_su2 = cosine_u1_u2(shared, u2)
        results['acos_su1'].append(acos_su1)
        results['grass_su1'].append(grassmann_distance(shared, u1, k=10))
        results['acos_su2'].append(acos_su2)
        results['grass_su2'].append(grassmann_distance(shared, u2, k=10))
        
    mean_acc= np.mean(results['accuracies'])
    std_acc= np.std(results['accuracies'])
    print(f"mean lin probing accuracy on {dataset} = {100 * mean_acc:.1f}")
    print(f"std of lin probing accuracy on {dataset} = {100 * std_acc:.2f}")   
    print(f"mean epoch time = {np.mean(results['epoch_times']):.1f}s ± {np.std(results['epoch_times']):.2f}s")
    for k in ['acos_u1u2', 'acos_su1', 'acos_su2','grass_u1u2', 'grass_su1', 'grass_su2']:
            arr = np.array(results[k])
            print(f"mean {k} = {arr.mean():.4f} ± {arr.std():.4f}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                        help='name of dataset, "mosi", "mosei", "humor", "sarcasm", "mimic"')
    
    args = parser.parse_args()
    assert args.dataset in ["mosi", "mosei", "humor", "sarcasm", "mimic"], "Dataset must be one of the following:'mosi',' 'mosei', 'humor', 'sarcasm', 'mimic' "
    main(args.dataset)
        

    
