# Define initial input image
input_path=data/sat_image.jpg

# Iterate over yearly transition configuration files
for config_path in config/yearly/past-to-present/*
do
  filename="$(basename -s .yaml $config_path).npy"
  output_path=data/predictions/past-to-present/$filename
  python run_image_distortion.py --cfg=$config_path --input=$input_path --o=$output_path
  input_path=$output_path
done
