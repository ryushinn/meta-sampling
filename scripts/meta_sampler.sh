# 1. specify the paths
# ----------
SCRIPT_PATH="src/meta_sampler.py"
EXP_PATH="results/"
DATA_PATH="data/brdfs/"
MODEL_PATH="data/meta-models/"
SAMPLER_PATH="data/meta-samplers/"

# 2. choose the model to be meta-trained
# ----------
model='nbrdf'
# model='phong'
# model='cooktorrance'

# hyperparameters
shots_list=(1 2 4 8 16 32 64 128 256 512)
k_list=(20)
meta_bs=1
fast_lr=1e-3
sampler_lr=5e-4
n_epochs=500
n_disp_ep=5

for k in "${k_list[@]}"; do
    for shots in "${shots_list[@]}"; do
        printf "meta-train %s samples in %s epochs for %s model\n" $shots $n_epochs $model
        python $SCRIPT_PATH --data_path $DATA_PATH --exp_path $EXP_PATH --save \
            --model_path $MODEL_PATH --sampler_path $SAMPLER_PATH \
            --model $model \
            --k $k --shots $shots --n_det $shots \
            --meta_bs $meta_bs \
            --fast_lr $fast_lr \
            --sampler_lr $sampler_lr \
            --n_epochs $n_epochs \
            --n_disp_ep $n_disp_ep
    done
done

