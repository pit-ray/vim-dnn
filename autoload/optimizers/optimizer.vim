vim9script

import autoload 'autograd.vim' as ag

var Tensor = ag.Tensor


export interface HasOptimizer
  this.params: list<Tensor>
  def Step()
  def ZeroGrad()
endinterface
