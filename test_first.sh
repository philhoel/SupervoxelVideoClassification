sbatch train_short_a100.slurm --num_heads 4 --hidden_features 512 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 5 --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_10381.pth --evaluate

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_126f6.pth --evaluate

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_98b8.pth --evaluate

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 7 --space_patch 9 --time_patch 4 --weight_init kaiming --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_b6d2.pth --evaluate

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 256 --dropout 0.2 --lr 1e-3 --maxlvl 6 --space_patch 10 --time_patch 3 --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_6d52.pth --evaluate

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 1024 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 3 --weight_init kaiming --epochs 100 --batch_size 32 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_a40a.pth --evaluate

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 512 --dropout 0.2 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 5 --weight_init xavier --epochs 100 --batch_size 32 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_389c.pth --evaluate

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 512 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 5 --space_lvl 2 --time_lvl 4 --sv alt --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_cdb9.pth --evaluate

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --space_lvl 2 --time_lvl 3 --sv alt --weight_init xavier --epochs 100 --batch_size 32 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_547d.pth --evaluate

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --space_lvl 2 --time_lvl 3 --sv alt --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_c1a6.pth --evaluate

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 7 --space_patch 9 --time_patch 4 --space_lvl 3 --time_lvl 4 --sv alt --weight_init kaiming --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_125e4.pth --evaluate

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 256 --dropout 0.2 --lr 1e-3 --maxlvl 6 --space_patch 10 --time_patch 3 --space_lvl 3 --time_lvl 3 --sv alt --weight_init xavier --epochs 100 --batch_size 64 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_.pth --evaluate

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 1024 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 3 --space_lvl 2 --time_lvl 4 --sv alt --weight_init kaiming --epochs 100 --batch_size 32 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_.pth --evaluate

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 512 --dropout 0.2 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 5 --space_lvl 2 --time_lvl 3 --sv alt --weight_init xavier --epochs 100 --batch_size 32 --data_workers 16 --pre_fetch 5 --dataset 1 --load_model id_.pth --evaluate

