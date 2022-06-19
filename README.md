# PolkaDemon
### Product counter (Re-id)
<br>
Для запуска программы нужно установить нужные для работы зависимости:

- Если у вас установлены NVIDIA CUDA - v11.2 и CUDAnn - v8.1.x, то вы можете использовать быстрые вычисления с помощью ГП GPU:
    

    pip intsall -r requirements.txt


---
## Использование

Запуск программы:
    
    python main.py

Результат выполнения:

>Время выполнения: 0:04:59.248271<br>
>Результат сохранён в submit_m1.0.csv

---
## Модель

Мы используем нейросеть MobileNet обученную на imagenet как классификатор с дополнительным полносвязным слоем, получая векторное представление изображения программа сравнивает их при помощи косинусного расстояния.


| Layer (type)                                             | Output |    ShapeParam |
|----------------------------------------------------------|:------:|--------------:|
| input_6 (InputLayer)                                     | \[(None, 224, 224, 3)] |  0 |  
| rescaling_2 (Rescaling)                                  | (None, 224, 224, 3) |  0 |     
| mobilenet_1.00_224 (Functional)                          | (None, 7, 7, 1024) |      3228864 |
 | global_average_pooling2d_2 <br>(GlobalAveragePooling2D) |  (None, 1024)      |       0    |
 | dropout_2 (Dropout)                                      |     (None, 1024)      |        0       |
 | dense_2 (Dense)                                          |   (None, 1024)         |     1049600   |

Total params: `4,278,464`<br>
Trainable params: `1,049,600`<br>
Non-trainable params: `3,228,864`<br>
_________________________________________________________________
