vim9script

import './nn/functional.vim' as F
export var ReLU = F.ReLU
export var Softmax = F.Softmax
export var CrossEntropyLoss = F.CrossEntropyLoss

import './nn/linear.vim'
export var Linear = linear.Linear

import './nn/module.vim'
export var HasModule = module.HasModule

import './optimizers/optimizer.vim'
export var HasOptimizer = optimizer.HasOptimizer

import './optimizers/sgd.vim'
export var SGD = sgd.SGD

defcompile
