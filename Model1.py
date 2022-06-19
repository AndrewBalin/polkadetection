import csv
from datetime import datetime
import json
import numpy as np
import torch.nn as nn
import tensorflow as tf
import cv2
import torch
import matplotlib as plt
from PIL import Image
from keras.models import load_model

start_time = datetime.now()

model = load_model('50k.h5')

model.summary()

dataset = 'PublicTestSet/'
th = 0.99676

def get_vec_lib(img):
    return model.predict(img)


def desk_to_cnt(bboxes, prod_img_path, desk_img_path, get_vec, th=th):
    cnt = 0
    img = tf.io.read_file(prod_img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[224, 224])
    img = tf.expand_dims(img, axis=0)
    prod_vec = get_vec(img)
    desk_img = cv2.imread(desk_img_path)
    for box in bboxes:
        img = desk_img[box[1]:box[3], box[0]:box[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = tf.expand_dims(img, axis=0)
        bbox_vec = get_vec(img)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        prod_vec = torch.as_tensor(np.array(prod_vec).astype('float'))
        bbox_vec = torch.as_tensor(np.array(bbox_vec).astype('float'))
        sim = cos(prod_vec, bbox_vec)
        print(sim)
        if sim >= th:
            cnt += 1

    return cnt


with open(dataset + 'shelves_info.json', 'r') as json_file:
    shelves = json.load(json_file)

with open(dataset + 'requests.csv', newline='') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

res = []
for pair in data:
    bboxes = shelves.get(pair[0])['bboxes']
    cnt = desk_to_cnt(bboxes, dataset + 'queries/' + pair[1], dataset + 'shelves/' + pair[0], get_vec_lib, th)
    res.append([pair[0], pair[1], cnt])

with open("submit.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(res)

print(f"\n\n\nВребя выполнения: {datetime.now() - start_time}")