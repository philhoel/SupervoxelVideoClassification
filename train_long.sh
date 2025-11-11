sbatch train_short_a100.slurm --num_heads 4 --hidden_features 512 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 5 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_7041.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_cc3a.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_acb4.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 7 --space_patch 9 --time_patch 4 --weight_init kaiming --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_189d.pth

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 256 --dropout 0.2 --lr 1e-3 --maxlvl 6 --space_patch 10 --time_patch 3 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_1563f.pth

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 1024 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 3 --weight_init kaiming --epochs 50 --batch_size 8 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_10d79.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 512 --dropout 0.2 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 5 --weight_init xavier --epochs 50 --batch_size 8 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_d1e5.pth



sbatch train_short_a100.slurm --num_heads 4 --hidden_features 512 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 5 --space_lvl 4 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_436a.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --space_lvl 3 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_dc3b.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --space_lvl 3 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_6375.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 7 --space_patch 9 --time_patch 4 --space_lvl 4 --time_lvl 3 --sv alt --weight_init kaiming --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_1e43.pth

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 256 --dropout 0.2 --lr 1e-3 --maxlvl 6 --space_patch 10 --time_patch 3 --space_lvl 3 --time_lvl 3 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_de3b.pth

sbatch train_short_a100.slurm --num_heads 4 --hidden_features 1024 --dropout 0.1 --lr 1e-3 --maxlvl 6 --space_patch 8 --time_patch 3 --space_lvl 4 --time_lvl 2 --sv alt --weight_init kaiming --epochs 50 --batch_size 8 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_16961.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 512 --dropout 0.2 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 5 --space_lvl 3 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 8 --data_workers 16 --pre_fetch 5 --dataset 3 --load_model id_873c.pth



sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 2 --load_model id_16c.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 2 --load_model id_10fac.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 7 --space_patch 9 --time_patch 4 --weight_init kaiming --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 2 --load_model id_30af.pth

sbatch train_short_a100.slurm --num_heads 8 --hidden_features 256 --dropout 0.3 --lr 1e-4 --maxlvl 5 --space_patch 12 --time_patch 3 --space_lvl 3 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 2 --load_model id_15193.pth

sbatch train_short_a100.slurm --num_heads 6 --hidden_features 256 --dropout 0.3 --lr 1e-3 --maxlvl 5 --space_patch 10 --time_patch 3 --space_lvl 3 --time_lvl 2 --sv alt --weight_init xavier --epochs 50 --batch_size 16 --data_workers 16 --pre_fetch 5 --dataset 2 --load_model id_10d92.pth

