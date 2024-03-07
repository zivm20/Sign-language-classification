import tensorflow as tf
from model_params import *
from keras.preprocessing.image import image_dataset_from_directory


@tf.function
def rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = (image / 255.0)
  return image, label



@tf.function
def random_transformations(image,label):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=(10.0)/(255.0), dtype=tf.float32)
    image = tf.image.random_flip_left_right(image)+ noise
    image = tf.image.random_hue(image,0.5) 
    image = tf.image.random_saturation(image,0.5,1.5)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.random_brightness(image,0.2)
    image = tf.image.random_contrast(image,0.5,1.5)
    image = tf.clip_by_value(image, 0, 1)

    

    return image,label




@tf.function
def load_resize_image(filename,label,img_dim=IMG_DIM):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (int(img_dim[0]), int(img_dim[1])))
    
    return image,label

# Function to load dataset from directory
def load_dataset(path:str,
                 train_val_split:tuple=TRAIN_VAL_SPLIT, 
                 img_dim:tuple=INPUT_SHAPE, 
                 seed:int=None,
                 batch_size:int=BATCH_SIZE,
                 )->tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset,list]:
    
    
    assert(len(train_val_split)==2)
    assert(train_val_split[0]+train_val_split[1] < 1.0)

    train_ds = image_dataset_from_directory(path,shuffle=True,
                                           validation_split=2*train_val_split[1],
                                           subset='training',
                                           image_size=(img_dim[0], img_dim[1]),
                                           batch_size=None,
                                           seed=seed)
    
    dataset_val_test = image_dataset_from_directory(path,shuffle=True,
                                           validation_split=2*train_val_split[1],
                                           subset='validation',
                                           image_size=(img_dim[0], img_dim[1]),
                                           batch_size=None,
                                           seed=seed)
    
    
    print("Splitting validation into validation and test")
    
    val_size = len(dataset_val_test)//2
    test_size = len(dataset_val_test) - val_size
    print("Using "+str(val_size)+" files for validation and "+str(test_size)+" files for test")
    
    
    val_ds = dataset_val_test.take(val_size)
    test_ds = dataset_val_test.skip(val_size).take(test_size)

    
    classNames = train_ds.class_names
    train_ds = train_ds.map(rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    val_ds = val_ds.map(rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size,num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    test_ds = test_ds.map(rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size,num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    
    return train_ds, val_ds, test_ds, classNames

    

    
    