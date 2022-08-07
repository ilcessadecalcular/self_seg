
python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --batch_size 1 \
    --crop_size 250 \
    --epochs 400 \
    --warmup_epochs 100 \
    --blr 1.5e-3 \
    --dist_eval \
    --weight_decay 0.05 \