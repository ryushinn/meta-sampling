# 1. specify the paths
# ----------
SCRIPT_PATH="src/meta_sampler_PCARR.py"
EXP_PATH="results/"
DATA_PATH="data/brdfs/"
MODEL_PATH="data/meta-models/"
SAMPLER_PATH="data/meta-samplers/"

# hyperparameters
n_det_lists=(1 2 4 8 16 32 64 128 256 512)
meta_bs=1
sampler_lr=1e-3
n_epochs=500
n_disp_ep=5

for n_det in "${n_det_lists[@]}"; do
    printf "meta-train %s samples in %s epochs for PCA model\n" $n_det $n_epochs
    python $SCRIPT_PATH --data_path $DATA_PATH --exp_path $EXP_PATH --save \
        --model_path $MODEL_PATH --sampler_path $SAMPLER_PATH \
        --n_det $n_det  \
        --meta_bs $meta_bs \
        --sampler_lr $sampler_lr \
        --n_epochs $n_epochs \
        --n_disp_ep $n_disp_ep
done

