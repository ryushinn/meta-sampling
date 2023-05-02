MERL_DATA_URL='https://www.dropbox.com/sh/yjt3bczfy52gb7o/AADvG_FhncJL59HgGOKxbE7Ya/brdfs?dl=1'
META_MODELS_URL='https://drive.google.com/uc?export=download&id=1AkHjQhPSo7QDTBaPhrI9uHdP2s_u7QYo'
META_SAMPLERS_URL='https://drive.google.com/uc?export=download&id=1NQ_ZVF5dQnFdFALKlipkYbNRj_MQwa3P'

wget -c $META_MODELS_URL -O data/meta-models/meta-models.zip \
&& unzip data/meta-models/meta-models.zip -d data/meta-models/ \
&& rm data/meta-models/meta-models.zip \
&& printf '=====Successfully download pretrained meta models=====\n'

wget -c $META_SAMPLERS_URL -O data/meta-samplers/meta-samplers.zip \
&& unzip data/meta-samplers/meta-samplers.zip -d data/meta-samplers/ \
&& rm data/meta-samplers/meta-samplers.zip \
&& printf '=====Successfully download trained meta samplers=====\n'

wget -c $MERL_DATA_URL -O data/brdfs/brdfs.zip \
&& unzip data/brdfs/brdfs.zip -x / -d data/brdfs/ \
&& rm data/brdfs/brdfs.zip \
&& printf '=====Successfully download MERL BRDF dataset=====\n'