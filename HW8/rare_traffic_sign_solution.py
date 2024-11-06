# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# !Этих импортов достаточно для решения данного задания


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = []
        for folder in root_folders:
            class_idx = self.class_to_idx[os.path.basename(folder)]
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                self.samples.append((img_path, class_idx))
        self.classes_to_samples = {}
        for class_idx in self.class_to_idx.values():
            self.classes_to_samples[class_idx] = [i for i, (_, idx) in enumerate(self.samples) if idx == class_idx]
        ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.transform = torchvision.transforms.Compose(
            torchvision.ToTensorV2(),
        )

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        return (Image.open(self.samples[index][0]), self.samples[index][0], self.samples[index][1])

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.jsonx
        """
        with open(path_to_classes_json, 'r') as json_file:
            json_dict = json.load(json_file)
        class_to_idx = {class_name: json_dict[class_name]["idx"] for class_name in json_dict}
        classes = {class_idx: class_name for class_name, class_idx in class_to_idx.items()}
        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        ### YOUR CODE HERE - список путей до картинок
        self.samples = ...
        ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.transform = ...
        self.targets = None
        if annotations_file is not None:
            ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            self.targets = ...

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        ### YOUR CODE HERE


class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = 1024,
    ):
        super().__init__()
        ### YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        ### YOUR CODE HERE

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        return  ### YOUR CODE HERE


def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    ### YOUR CODE HERE
    return model


def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = ...
    return results


def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
        ### YOUR CODE HERE

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = ...
        ### YOUR CODE HERE - случайное изображение фона
        bg = ...
        return  ### YOUR CODE HERE


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        ### YOUR CODE HERE

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """
        ### YOUR CODE HERE


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        ### YOUR CODE HERE

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        ### YOUR CODE HERE


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    ### YOUR CODE HERE
    return model


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.eval()
        ### YOUR CODE HERE

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        ### YOUR CODE HERE

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        ### YOUR CODE HERE

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        ### YOUR CODE HERE - предсказание нейросетевой модели
        features, model_pred = ...
        features = features / np.linalg.norm(features, axis=1)[:, None]
        ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = ...
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        return  ### YOUR CODE HERE

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        ### YOUR CODE HERE


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE


if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.
    pass
