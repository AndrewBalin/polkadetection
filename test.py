import random
from datetime import datetime
import os
import torch.nn as nn
from img2vec_pytorch import Img2Vec
from PIL import Image

cuda = True

start_time = datetime.now()

def init():
    # Model initialization
    return Img2Vec(cuda=cuda, model='vgg', layer=3, layer_output_size=4096)


def img_to_vec(img, model):
    # Transforming
    return model.get_vec(img, tensor=True)


model = init()

dataset = '../Data/train_dataset_1/'
n = 20

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
a = []
for _ in range(10):
    rand_products = random.sample(range(0, 2388), n)
    for i in rand_products:
        for j in range(len(os.listdir(dataset + str(i))) // 2):
            try:
                img1 = Image.open(dataset + str(i) + '/' + str(2 * j) + '.jpg')
                img2 = Image.open(dataset + str(i) + '/' + str(2 * j + 1) + '.jpg')
                a.append(cos(img_to_vec(img1, model), img_to_vec(img2, model)))
            except:
                continue

print(sum(a) / len(a))
print(f"\n\n\nВребя выполнения: {datetime.now() - start_time}, CUDA: {cuda}")