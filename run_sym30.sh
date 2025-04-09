
nohup python ./src/main.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_TW_sym30_ucr128.csv --TW_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 0 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 128 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --soft_interpre_dim label \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &

nohup python ./src/main.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_ori_sym30_ucr128.csv --ori_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 0 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 128 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --soft_interpre_dim label \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &

nohup python ./src/main.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_TW_sym30.csv --TW_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 0 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 2 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --soft_interpre_dim label \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &

nohup python ./src/main_pt.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_TW_sym30_two.csv --TW_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 1 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 --manual_seeds 19 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 2 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --soft_interpre_dim label \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &

nohup python ./src/main.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_ori_sym30.csv --ori_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 0 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 2 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &

nohup python ./src/main_pt.py --model SALA --epochs 200 --warmup 10 --lr 1e-3 \
--label_noise 0 --outfile clean_ori_sym30_two.csv --ori_clean_sample --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 1 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 --manual_seeds 19 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 2 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --soft_interpre_dim label \
--only_max_min_noise max_min --only_max_min_mask max_min >/dev/null 2>&1 &