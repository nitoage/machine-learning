# -*- coding: utf-8 -*-

import os.path
import pickle
from chainer import optimizers, serializers

class ModelUtil:
    def __init__(self):
        '''
        初期設定
        '''
        self.model_pkl = './mnist_sample.pkl'
        self.model_name = './mnist_sample.model'
        self.optimizer_name = './mnist_sample.state'
        
    def set_model_pkl(self, model_pkl):
        self.model_pkl = model_pkl

    def get_model_pkl(self):
        return self.model_pkl
    
    def set_model_name(self, model_name):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    def set_optimizer_name(self, optimizer_name):
        self.optimizer_name = optimizer_name

    def get_optimizer_name(self):
        return self.optimizer_name
    
    def dump_model_and_optimizer(self, model, optimizer):
        '''
        modelとoptimizerを保存
        '''
        print('save the model')
        serializers.save_npz(self.model_name, model)
        print('save the optimizer')
        serializers.save_npz(self.optimizer_name, optimizer)

    def load_model_and_optimizer(self):
        '''
        modelとoptimizerをロード
        '''
        if os.path.exists(self.model_name):
            model = serializers.load_npz(self.model_name, model)
        if os.path.exists(self.optimizer_name):
            optimizer = serializers.load_npz(self.optimizer_name, optimizer)
        return model, optimizer


    def dump_model(self, model):
        '''
        modelを保存
        '''
        with open(self.model_pkl, 'wb') as pkl:
            pickle.dump(model, pkl, -1)

    def load_model(self):
        '''
        modelを読み込む
        '''
        model = None
        if os.path.exists(self.model_pkl):                
            with open(self.model_pkl, 'rb') as pkl:
                model = pickle.load(pkl)
        return model