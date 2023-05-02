# 1. specify the paths
# ----------
SCRIPT_PATH="src/train_model.py"
EXP_PATH="results/"
DATA_PATH="data/brdfs/"

# 2. choose the BRDF to be trained
# ----------
BRDF_names=(alumina-oxide)

# 3. choose the right mode
# ----------
mode='overfit'
# mode='classic'

# hyperparameters
if [[ $mode == 'overfit' ]]; then
    bs_list=(512)
    n_iter=50000
    lr=5e-4
elif [[ $mode == 'classic' ]]; then
    bs_list=(1 2 4 8 16 32 64 128 256 512)
    n_iter=20
    lr=1e-3
else
    printf "WRONG MODE!"
fi

# 4. choose the model class to be fitted
# ----------
model='nbrdf'
# model='phong'
# model='cooktorrance'

for bs in "${bs_list[@]}"; do
    for name in "${BRDF_names[@]}"; do
        printf "Mode %s: fit %s to BRDF (%s) in %s iterations with %s batchsize\n" $mode $model $name $n_iter $bs
        python $SCRIPT_PATH --data_path $DATA_PATH --exp_path $EXP_PATH --save\
            --mode $mode \
            --model $model \
            --brdf_name $name \
            --batch_size $bs \
            --n_iter $n_iter \
            --lr $lr
    done
done