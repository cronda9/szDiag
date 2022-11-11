import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFBertModel

def generate_training_data(df, ids, masks, tokenizer):
  for i, text in tqdm(enumerate(df['sentence'])):
    tokenized_text = tokenizer.encode_plus(
          text,
          max_length=512,
          truncation=True,
          padding='max_length',
          add_special_tokens=True,
          return_tensors='tf')
    ids[i, :] = tokenized_text.input_ids 
    masks[i, :] = tokenized_text.attention_mask 
  return ids, masks


def SentimentDatasetMapFunction(input_ids, attention_masks, labels):
  return {
      'input_ids': input_ids,
      'attention_mask': attention_masks
  }, labels

df = pd.read_csv('data.csv')
df_s = df[df['disease'] == 'Schizophrenia']
df_bi = df[df['disease'] == 'Bipolar']
df_sa = df[df['disease'] == 'Schizoaffective']
df_c = df[df['disease'] == 'Control']


df_s = df_s.sample(136)
df_bi = df_bi.sample(136)
df_sa = df_sa.sample(136)
df_c = df_c.sample(136)

df_balanced = pd.concat([df_s, df_bi, df_sa, df_c])
df_balanced = df_balanced.dropna()
label_dict = {'Schizophrenia': 0, 'Bipolar':1, 'Schizoaffective':2, 'Control': 3}
X_train, X_test, y_train, y_test = train_test_split(df_balanced['sentence'],df_balanced['id'], stratify=df_balanced['id'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_input_ids = np.zeros((len(df_balanced), 512))
X_attention_masks = np.zeros((len(df_balanced), 512))

X_input_ids, X_attention_masks = generate_training_data(df_balanced, X_input_ids, X_attention_masks, tokenizer)
labels = np.zeros((len(df_balanced), 4))
labels[np.arange(len(df_balanced)), df_balanced['id'].values] = 1
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attention_masks, labels))

dataset = dataset.map(SentimentDatasetMapFunction)
dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)
p = 0.8
train_size = int((len(df_balanced)//16)*p)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
attention_masks = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

bert_embds = bert_model.bert(input_ids, attention_mask=attention_masks)[1]
l = tf.keras.layers.Dropout(0.1, name="dropout")(bert_embds)
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(l)
output_layer = tf.keras.layers.Dense(4, activation='softmax', name='output_layer')(intermediate_layer)

model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output_layer)
model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
model.compile(optim, loss_func, acc)
hist = model.fit(train_dataset, validation_data=val_dataset,epochs=1)
