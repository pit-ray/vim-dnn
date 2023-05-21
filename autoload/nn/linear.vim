vim9script

import autoload 'autograd.vim' as ag
import './module.vim'

var Tensor = ag.Tensor
var HasModule = module.HasModule


export class Linear implements HasModule
  public this.weight: Tensor
  public this.bias: Tensor

  def new(in_channels: number, out_channels: number)
    var std = sqrt(2.0 / in_channels)
    this.weight = ag.Normal(0.0, std, [in_channels, out_channels])
    this.bias = ag.Zeros([out_channels])

    this.weight.SetName('weight')
    this.bias.SetName('bias')
  enddef

  def GetParameters(): list<Tensor>
    return [this.weight, this.bias]
  enddef

  def Forward(...inputs: list<any>): Tensor
    return ag.Add(ag.Matmul(inputs[0], this.weight), this.bias)
  enddef
endclass
