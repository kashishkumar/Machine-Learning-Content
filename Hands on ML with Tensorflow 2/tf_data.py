import tensorflow as tf
X=tf.range(10)
dataset=tf.data.Dataset.from_tensor_slices(X)
dataset
tf.data.Dataset.range(10)
a=dataset.repeat(3).batch(7)    
for item in a:
    print(item)
b=dataset.batch(7,drop_remainder=True).repeat(3)
for item in b:
    print(item) 
data=dataset.map(lambda x:x**3)    
for item in data:
    print(item)
X**3
"""While the map() applies a transformation to each item, the apply() method applies a
transformation to the dataset as a whole."""
data=data.apply(tf.contrib.data.unbatch()) 
data=data.filter(lambda x: x<10)                # Filters value less than 10
for item in dataset.take(5):
    print(item)                                 # View just few items from the dataset
dataset=tf.data.Dataset.range(10).repeat(3)
for item in dataset:
    print(item)
dataset=dataset.shuffle(buffer_size=5,seed=42).batch(7) # Higher the shuffle size, better is the shuffling
for item in dataset:
    print(item)    

n_inputs = 8

def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

#Preprocessing 
def csv_reader_dataset(filepaths, repeat=None, n_readers=5,
    n_read_threads=None, shuffle_buffer_size=10000,
    n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)