from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

target_size = (512, 512)
seed = np.random.randint(0,1e5)

image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator()

test_image_generator = image_datagen.flow_from_directory('data/phyto/validation/val_imgs/', seed=seed, target_size=target_size, class_mode=None, batch_size = 6)
test_mask_generator = mask_datagen.flow_from_directory('data/phyto/validation/val_labels/', seed=seed, target_size=target_size, class_mode=None, batch_size = 6)

def combine_generator(gen1, gen2, batch_list=6,training=True):
  
    while True:
        image_batch, label_batch=next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
        image_batch, label_batch=np.expand_dims(image_batch,axis=0),np.expand_dims(label_batch,axis=0)

        for i in range(batch_list-1):
            image_i,label_i = next(gen1)[0], np.expand_dims(next(gen2)[0][:,:,0],axis=-1)
            
            image_i, label_i=np.expand_dims(image_i,axis=0),np.expand_dims(label_i,axis=0)
            image_batch=np.concatenate([image_batch,image_i],axis=0)
            label_batch=np.concatenate([label_batch,label_i],axis=0)
              
        yield((image_batch,label_batch))

test_generator = combine_generator(test_image_generator, test_mask_generator,training=True)


def show_predictions_in_test(model_name, generator=None, num=3):
    if generator ==None:
        generator = test_generator
    for i in range(num):
        image, mask=next(generator)
        sample_image, sample_mask= image[1], mask[1]
        image = np.expand_dims(sample_image, axis=0)
        pr_mask = model_name.predict(image)
        pr_mask=np.expand_dims(pr_mask[0].argmax(axis=-1),axis=-1)
        display([sample_image, sample_mask,pr_mask])
        
def display(display_list,title=['Input Image', 'True Mask', 'Predicted Mask']):
    plt.figure(figsize=(15, 15))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),cmap='magma')
        plt.axis('off')
    plt.show()