import input_pipeline.TFRecord as tfr

data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset"

path, filenames = tfr.prepare_images(data_dir, 'train')
print(filenames)

filenames.sort(key=lambda x: int(x[-7:-4]))
print(filenames)

