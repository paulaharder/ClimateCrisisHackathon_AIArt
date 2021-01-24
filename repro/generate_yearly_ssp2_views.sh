# Define initial input image
input_path=data/predictions/past-to-present/2013-to-2014.npy

# Iterate over yearly transition configuration files
for config_path in config/yearly/SSP2/*
do
  filename="$(basename -s .yaml $config_path).npy"
  output_path=data/predictions/SSP2/$filename
  python run_image_distortion.py --cfg=$config_path --input=$input_path --o=$output_path
  input_path=$output_path
done
