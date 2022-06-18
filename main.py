import torch.nn as nn
from img2vec_pytorch import Img2Vec
from PIL import Image
import cv2
import csv
import json

img2vec = Img2Vec(cuda=True)

dataset = 'PublicTestSet/'
th = 0.85


def get_vec_lib(img):
    return img2vec.get_vec(img, tensor=True)


def desk_to_cnt(bboxes, prod_img_path, desk_img_path, get_vec, th=0.85):
    cnt = 0
    prod_vec = get_vec(Image.open(prod_img_path))
    desk_img = cv2.imread(desk_img_path)
    for box in bboxes:
        img = desk_img[box[1]:box[3], box[0]:box[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        bbox_vec = get_vec(im_pil)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(prod_vec, bbox_vec)
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
