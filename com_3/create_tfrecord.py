import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import pandas as pd
import tensorflow as tf
import _pickle as cPickle
from sklearn.model_selection import train_test_split


# create tfrecords for dataset 

def create_tfrecords(df_cap, img_df, filename, num_files=5):
    ''' create tfrecords for dataset '''
    
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    num_records_per_file = img_df.shape[0] // num_files
    
    total_count = 0  
    
    print("create training dataset....")
    for i in range(num_files):
        # tfrecord writer: write record into files
        count = 0
        writer = tf.python_io.TFRecordWriter(filename+ '-' + str(i+1) + '.tfrecord')

        # put remaining records in last file
        st = i * num_records_per_file                                              # start point (inclusive)
        ed = (i+1) * num_records_per_file if i != num_files-1 else img_df.shape[0] # end point (exclusive)

        for idx, row in img_df.iloc[st : ed].iterrows():
            
            img_representation = row['img']                 # img representation in 256-d array format

            # each image has some captions describing it.
            for _, inner_row in df_cap[df_cap['img_id'] == row['img_id']].iterrows():
                caption = eval(inner_row['caption'])        # caption in different sequence length list format

                # construct 'example' object containing 'img', 'caption' 
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': _float_feature(img_representation),
                    'caption': _int64_feature(caption)
                }))

                count += 1
                writer.write(example.SerializeToString())
        print("create {}-{}.tfrecord -- contains {} records".format(filename, str(i+1), count))
        total_count += count
        writer.close()
    print("Total records: {}".format(total_count))

df_cap = pd.read_csv('dataset/text/train_enc_cap.csv') # a dataframe - 'img_id', 'cpation'
img_train = cPickle.load(open('dataset/train_img256.pkl', 'rb')) # a dictionary - keys: 'img_id', values: '256-d array
# transform img_dict to dataframe
img_train_df = pd.DataFrame(list(img_train.items()), columns=['img_id', 'img'])

create_tfrecords(df_cap, img_train_df, 'dataset/tfrecord/train', 10)