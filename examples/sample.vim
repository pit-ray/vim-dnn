vim9script

import autoload 'autograd.vim' as ag
import '../autoload/dnn.vim' as nn

var Tensor = ag.Tensor
var HasModule = nn.HasModule


class MLP implements HasModule
  this.l1: HasModule
  this.l2: HasModule

  def new(in_size: number, classes: number)
    this.l1 = nn.Linear.new(in_size, 100)
    this.l2 = nn.Linear.new(100, classes)
  enddef

  def GetParameters(): list<Tensor>
    var params: list<Tensor>
    params += this.l1.GetParameters()
    params += this.l2.GetParameters()
    return params
  enddef

  def Forward(...inputs: list<any>): Tensor
    var h = nn.ReLU(this.l1.Forward(inputs[0]))
    h = nn.Softmax(this.l2.Forward(h))
    return h
  enddef
endclass


def Main()
  var model = MLP.new(3, 13)

  var inputs = ag.Zeros([2, 3])
  var y = model.Forward(inputs)
  echo y.data
enddef


Main()
