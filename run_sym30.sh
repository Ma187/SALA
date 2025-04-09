nohup python ./src/main.py --model SALA --epochs 200 --warmup 20 --lr 1e-3 \
--label_noise 0 --ni 0.3 --outfile SALA_sym30.csv --num_workers 0 \
--cuda_device 0 --valid_set --MILLET --pseudo_loss_coef 1. --amp_noise 0.1 \
--patch --forward_type pm --amp_mask 0.1 --ucr 128 --recon --L_rec_coef 0.1 \
--consistency_loss_coef 0.1 --only_max_min_noise max_min \
--only_max_min_mask max_min >/dev/null 2>&1 &