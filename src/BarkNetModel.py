import io, os
import json
import torch, torchvision
from torch.autograd import Variable
from torchvision.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import cv2
import numpy as np

class BarkNet:

    def __init__(self, model_path='log/config/0/0'):
        print(' * createing BarkNet...')

        self.model_path = model_path
        self.species = {}
        self.classes = []
        self.net = []
        self.wiki = 'https://en.wikipedia.org/wiki/'

        self.get_classes()
        self.load_model()

        print(' * BarkNet created.')
        

    def get_classes(self):
        species = {
        'BOJ': 'Betula alleghaniensis',
        'BOP': 'Betula papyrifera',
        'CHR': 'Quercus rubra',
        'EPB': 'Picea glauca',
        'EPN': 'Picea mariana',
        'EPO': 'Picea abies',
        'EPR': 'Picea rubens',
        'ERB': 'Acer platanoides',
        'ERR': 'Acer rubrum',
        'ERS': 'Acer saccharum',
        'FRA': 'Fraxinus americana',
        'HEG': 'Fagus grandifolia',
        'MEL': 'Larix laricina',
        'ORA': 'Ulmus americana',
        'OSV': 'Ostrya virginiana',
        'PEG': 'Populus grandidentata',
        'PET': 'Populus tremuloides',
        'PIB': 'Pinus strobus',
        'PID': 'Pinus rigida',
        'PIR': 'Pinus resinosa',
        'PRU': 'Tsuga canadensis',
        'SAB': 'Abies balsamea',
        'THO': 'Thuja occidentalis'
        }
        self.species = species
        self.classes = [*species]


    def load_model(self):
        self.net = torch.load(self.model_path)
        self.net.eval()
        print(' * CNN model loaded.')

    def convert_image(self, img):
        img = np.array(img)
        if len(img.shape) > 2 and img.shape[2] == 4:
            return Image.fromarray(img[...,:3])
        else: 
            return img

    def split_crops(self, img):
        CROP_SIZE = 224
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
        predictions = output.max(1)[1]
        predictions = predictions.cpu()
        flat_results = predictions.tolist()
        return flat_results


    def get_most_probable_species(self, counter, length):
        list_of_species = []
        arr_best_occs = dict(counter.most_common(3)).keys()
        my_dict = dict(counter)

        for k in my_dict:
            if k in arr_best_occs and my_dict[k]/length*100 > 30:
                code = self.classes[k]
                plant_name = self.species[code]
                list_of_species.append(plant_name)

        return list_of_species


    def return_classification_array(self, file_in):
        img = Image.open(file_in)
        img = self.convert_image(img)
        crops = self.split_crops(img)
        if len(crops) > 0:
            with torch.no_grad():
                inp = Variable(crops)
                output = self.net(inp)

                flat_results = self.get_class_predictions(output)
                # get 3 most common keys of dict
                cnt_occurencess = Counter(flat_results)
                most_probable_species = self.get_most_probable_species(cnt_occurencess, len(flat_results))

                main_pred = max(set(flat_results), key=flat_results.count)
                
                website = self.wiki + most_probable_species[0].replace(" ", "_")

                print(' * occs:\t', most_probable_species)
                print(' * pred:\t', most_probable_species[0])
                print(' * webs:\t', website)

                final_dict = {'pred': most_probable_species[0], 'predtable' : most_probable_species, 'website': website}
                return final_dict

