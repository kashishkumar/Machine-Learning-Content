import glob
import os
import tensorflow as tf
import DCGANs
#path="/home/test/Desktop/Projects/GANs - Bollywood/IMFDB_final/**/**/*.jpg"
data_dir="/home/test/Desktop/Projects/GANs - Bollywood/IMFDB_final"
#configfiles = glob.glob(path,recursive=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE=32
IMG_HEIGHT=72
IMG_WIDTH=72
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    )

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
        
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch) 



