from torch.autograd import Variable
from torchvision.transforms import *
import torch
import os
import json
import math
from PIL import Image
import numpy as np

CROP_SIZE = 224
import time

class Test:

    def __init__(self, model_name, model_path=None, log_path=None, dataset=None, multitask=True):
        self.model_name = model_name
        self.log_path = log_path
        self.multitask=multitask

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = model_name

        self.net = None
        self.test_file = None

        self.dataset = dataset
        self.classes = []

        self._load_classes()
        self.classes.sort()
        print('classes sorted:', self.classes)
        self._create_network()

    def run(self, test_file_name):
        print('running')
        self._create_test_file(test_file_name)
        # for f in self.dataset['files']:
        #     print('iteration')
        # f = '../data/41_CHR_83_GalaxyS5_20170607_134920_10.jpg'
        # f = '../data/169_EPO_69_Nexus 5_20170922_102428_7.jpg'
        f = '../data/695_EPN_82_GalaxyS5_20170622_100334_10.jpg'
        
        class_name = f.split('/')[2]
        print('classname', class_name)
        img = Image.open(f)
        crops = self.split_crops(img)

        if len(crops) > 0:
            print('has crops')
            with torch.no_grad():
                inp = Variable(crops)
                # input = Variable(crops, volatile=True).cuda()
                # print(inp)
                output = self.net(inp)
                print('output', output)
                pred = self.get_class_predictions(output)
                # pred2 = self._get_specific_predictions(output, self.classes)
                print('get prediction')
                if self.multitask:
                    dbh = self.get_dbh_predictions(output)
                else:
                    dbh = 0

                print(class_name, pred)
                # self.write_results(class_name, f, pred, dbh)

    def run_single_crop(self, test_file_name, batch_size=32):
        self._create_test_file(test_file_name)
        batch_input = []
        batch_files = []
        test_size = len(self.dataset['files'])
        times = []
        for i, file in enumerate(self.dataset['files']):
            start = time.time()
            img = Image.open(file)
            crop = ToTensor()(RandomCrop(224)(img))

            batch_input.append(crop)
            batch_files.append(file)

            if (i+1) % batch_size == 0 or (i+1) == test_size:
                batch_input = Variable(torch.stack(batch_input), volatile=True)
                # batch_input = Variable(torch.stack(batch_input), volatile=True).cuda()
                output = self.net(batch_input)

                if self.multitask:
                    predictions = output[0].max(1)[1].cpu().data.numpy().tolist()
                    dbh_predictions = output[1].cpu().data.numpy().tolist()
                    dbh_predictions = [item for sublist in dbh_predictions for item in sublist]
                else:
                    predictions = output.max(1)[1].cpu().data.numpy().tolist()
                    dbh_predictions = np.zeros(batch_size)

                for j, prediction in enumerate(predictions):
                    file = batch_files[j]
                    class_name = file.split('/')[-2]
                    #self.write_results(class_name, file, pred=prediction, dbh=dbh_predictions[j])

                batch_input = []
                batch_files = []
                end = time.time()
                times.append(end - start)
                print(sum(times) / len(times))

    def _get_specific_predictions(self, output, specific_classes):
        results = []
        class_index = []
        for class_name in specific_classes:
            class_index.append(self.classes.index(class_name))

        for i, crop in enumerate(output[0]):
            results.append([])
            for index in class_index:
                results[i].append(crop[index].data[0])

        preds =[]
        for result in results:
            preds.append(class_index[result.index(max(result))])

        return max(set(preds), key=preds.count)

    def _load_classes(self):
        for file in self.dataset['files']:
            class_name = file.split('/')[-2]
            if class_name not in self.classes:
                self.classes.append(class_name)
        # print(self.classes)

    def _create_network(self):
        name = os.path.join(self.log_path, self.model_path, self.model_name)
        print('network-name', name)
        self.net = torch.load(name)
        # print(self.net)
        # self.net.cuda()
        self.net.eval()

    def _create_test_file(self, test_file_name):
        self.test_file = open(self.log_path + self.model_path + '/{}'.format(test_file_name), 'w', 1)
        for i, target in enumerate(self.classes):
            self.test_file.write(target)
            if i != len(self.classes):
                self.test_file.write(', ')
        self.test_file.write('\n')

    @staticmethod
    def split_crops(img):
        crops = []
        for i in range(img.size[1] // CROP_SIZE):
            for j in range(img.size[0] // CROP_SIZE):
                start_y = i * CROP_SIZE
                start_x = j * CROP_SIZE

                crop = img.crop((start_x, start_y, start_x + CROP_SIZE, start_y + CROP_SIZE))
                crop = ToTensor()(crop)
                crops.append(crop)

        if len(crops) > 0:
            return torch.stack(crops)
        else:
            return []

    def get_class_predictions(self, output):
        if self.multitask:
            output = output[0]
        predictions = output.max(1)[1]
        predictions = predictions.cpu()

        print('pred:data:', predictions.data)
        # predictions = predictions.data.numpy()
        # predictions = predictions.data.detach().numpy()
        flat_results = predictions.tolist()
        pred = max(set(flat_results), key=flat_results.count)
        return pred

    @staticmethod
    def get_dbh_predictions(output):
        dbh = torch.mean(output[1])
        dbh = dbh.data[0]
        return dbh

    def write_results(self, class_name, file_path, pred, dbh):
        self.test_file.write(
            '{}, {}, {}, {}\n'.format(file_path, self.classes.index(class_name),
                                      pred, dbh))
        print('{} - {}, {}'.format(self.classes.index(class_name), pred,
                                   math.fabs(int(file_path.split('/')[-1].split('_')[2]) / math.pi - dbh)))


if __name__ == '__main__':
    model = str(0)
    log_path = '../log/'

    model_path = 'config' + '/' + model

    dataset_file = os.path.join(log_path, model_path, 'dataset')
    dataset_file = open(dataset_file)
    loaded_dataset = json.load(dataset_file)
    dataset_file.close()

    test = Test(model, model_path, log_path, dataset=loaded_dataset['test'], multitask=False)
    test.run(test_file_name='test_run')
