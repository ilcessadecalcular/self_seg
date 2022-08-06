
python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --batch_size 1 \
    --crop_size 50 \
    --epochs 200 \
    --warmup_epochs 50 \
    --blr 1.5e-4 \
    --dist_eval \
    --weight_decay 0.05 \