from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

target_size = (512, 512)
batch_size = 2

image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator()

seed = np.random.randint(0,1e5)

test_mask_generator = mask_datagen.flow_from_directory('data/phyto/validation/val_labels',seed=seed, target_size=target_size,batch_size = batch_size)

test_image_generator = image_datagen.flow_from_directory('data/phyto/validation/val_imgs',seed=seed, target_size=target_size, batch_size = batch_size)

def combine_generator(gen1, gen2,batch_size=6,training=True):
    while True:
        image_batch, label_batch=next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
        image_batch, label_batch=np.expand_dims(image_batch,axis=0),np.expand_dims(label_batch,axis=0)

        for i in range(batch_size-1):
            image_i,label_i = next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
              
        yield((image_batch,label_batch))

test_generator = combine_generator(test_image_generator, test_mask_generator,training=True)


def show_predictions_in_test(generator=None, num=3):
    if generator ==None:
        generator = test_generator
    for i in range(num):
        image, mask=next(generator)
        print(image)
        sample_image, sample_mask= image[1], mask[1]
        image = np.expand_dims(sample_image, axis=0)

        pr_mask = model.predict(image)
        pr_mask=np.expand_dims(pr_mask[0].argmax(axis=-1),axis=-1)
        display([sample_image, sample_mask,pr_mask])