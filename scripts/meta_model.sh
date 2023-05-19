# 1. specify the paths
# ----------
SCRIPT_PATH="src/meta_model.py"
EXP_PATH="results/"
DATA_PATH="data/brdfs/"
MODEL_PATH="data/meta-models/"

# 2. choose the model to be meta-trained
# ----------
model='nbrdf'
# model='phong'
# model='cooktorrance'

# hyperparameters
k=20
shots=512
meta_bs=1
fast_lr=1e-3
meta_lr=1e-4
n_epochs=10000
n_disp_ep=100

printf "meta-train %s model in %s epochs\n" $model $n_epochs
python $SCRIPT_PATH --data_path $DATA_PATH --exp_path $EXP_PATH --save \
    --model_path $MODEL_PATH \
    --model $model \
    --k $k --shots $shots \
    --meta_bs $meta_bs \
    --fast_lr $fast_lr \
    --meta_lr $meta_lr \
    --n_epochs $n_epochs \
    --n_disp_ep $n_disp_ep

