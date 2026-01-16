import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
torch.cuda.empty_cache()
from pytorch_lightning import Trainer
from modules.coral import COrAL
from modules.mmfusion import MMFusion
from modules.alexnet import AlexNetEncoder
from dataset.trifeatures import TrifeaturesDataModule
from modules.input_adapters import PatchedInputAdapter
import warnings
from evaluation.linear_probe import evaluate_linear_probe


def classification_scoring(model, data_module):
    Z_train, y_train = model.extract_features(data_module.train_dataloader())
    Z_val, y_val = model.extract_features(data_module.val_dataloader())
    Z_test, y_test = model.extract_features(data_module.test_dataloader())
    scores = evaluate_linear_probe(Z_train, y_train, Z_test, y_test, Z_val, y_val)
    return scores["acc1"]

def main():
    results = {'accuracies_synergy': [],'accuracies_uniqueness': [],'accuracies_redundancy': []}

    for seed in range(41, 46):  
        print("seed:", seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed) 
        warnings.filterwarnings("ignore", category=UserWarning) 

        data_module_R_U = TrifeaturesDataModule("COrAL", "bimodal", batch_size=64, num_workers=4, biased=False)
        data_module_S = TrifeaturesDataModule("COrAL", "bimodal", batch_size=64, num_workers=4, biased=True)
    
        # The downstream tasks
        downstream_task_S = TrifeaturesDataModule("Sup","bimodal", batch_size=64, num_workers=4, biased=False, task="synergy")
        downstream_task_U1 = TrifeaturesDataModule("Sup","bimodal", batch_size=64, num_workers=4, biased=False, task="unique1")
        downstream_task_R = TrifeaturesDataModule("Sup","bimodal", batch_size=64, num_workers=4, biased=False, task="share")
    
        
        coral= COrAL(
            encoder=MMFusion(
                        encoders=[ # Symmetric visual encoders
                            AlexNetEncoder(latent_dim=256, dropout=0.5, global_pool=""), 
                            AlexNetEncoder(latent_dim=256, dropout=0.5, global_pool="")
                        ], 
                        input_adapters=[PatchedInputAdapter(num_channels=256, stride_level=1, patch_size_full=1, dim_tokens=512, image_size=6),
                            PatchedInputAdapter(num_channels=256, stride_level=1, patch_size_full=1, dim_tokens=512, image_size=6)], # No adapters needed
                        lin_layers=[nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256)),
                                    nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256))],
                        embed_dim=512
                    ),
                    projection=COrAL._build_mlp(512, 512, 256),
                    uni_projection=COrAL._build_mlp(256, 512,256),
                    optim_kwargs=dict(lr=1e-4, weight_decay=1e-3),
                    loss_kwargs=dict(temperature=0.1, ortho_weights=1),
                    asym_masking = True)

        trainer = Trainer(inference_mode=False, max_epochs=100)
        print("synergy")
        trainer.fit(coral, datamodule=data_module_S)
        score_acc= classification_scoring(coral, downstream_task_S)
        print(f"COrAL accuracy on synergy task trifeature lin probing={100 * score_acc:.1f}")
        results['accuracies_synergy'].append(score_acc)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed) 

        coral= COrAL(
            encoder=MMFusion(
                        encoders=[ # Symmetric visual encoders
                            AlexNetEncoder(latent_dim=256, dropout=0.5, global_pool=""), 
                            AlexNetEncoder(latent_dim=256, dropout=0.5, global_pool="")
                        ], 
                        input_adapters=[PatchedInputAdapter(num_channels=256, stride_level=1, patch_size_full=1, dim_tokens=512, image_size=6),
                            PatchedInputAdapter(num_channels=256, stride_level=1, patch_size_full=1, dim_tokens=512, image_size=6)], # No adapters needed
                        lin_layers=[nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256)),
                                    nn.Sequential( nn.Linear(512, 512),nn.LayerNorm(512),nn.GELU(), nn.Linear(512, 256))],
                        embed_dim=512
                    ),
                    projection=COrAL._build_mlp(512, 512, 256),
                    uni_projection=COrAL._build_mlp(256, 512,256),
                    optim_kwargs=dict(lr=1e-4, weight_decay=1e-3),
                    loss_kwargs=dict(temperature=0.1, ortho_weights=1),
                    asym_masking = True)

        trainer = Trainer(inference_mode=False, max_epochs=100)
        print("redundancy and uniqueness")
        trainer.fit(coral, datamodule=data_module_R_U)
        score_acc= classification_scoring(coral, downstream_task_U1)
        print(f"COrAL accuracy on uniqueness task trifeature lin probing={100 * score_acc:.1f}")
        results['accuracies_uniqueness'].append(score_acc)
        score_acc= classification_scoring(coral, downstream_task_R)
        print(f"COrAL accuracy on redundancy task trifeature lin probing={100 * score_acc:.1f}")
        results['accuracies_redundancy'].append(score_acc)


    mean_syn = np.mean(results['accuracies_synergy'])
    std_syn = np.std(results['accuracies_synergy'])
    mean_uni = np.mean(results['accuracies_uniqueness'])
    std_uni = np.std(results['accuracies_uniqueness'])
    mean_red = np.mean(results['accuracies_redundancy'])
    std_red = np.std(results['accuracies_redundancy'])

    print(f"overall COrAL accuracy on synergy={100 * mean_syn:.1f}")
    print(f"overall COrAL standard deviation on synergy={100 * std_syn:.2f}")

    print(f"overall COrAL accuracy on uniqueness={100 * mean_uni:.1f}")
    print(f"overall COrAL standard deviation on uniqueness={100 * std_uni:.2f}")

    print(f"overall COrAL accuracy on redundancy={100 * mean_red:.1f}")
    print(f"overall COrAL standard deviation on redundancy={100 * std_red:.2f}")
        
if __name__ == "__main__":
    main()
        

    