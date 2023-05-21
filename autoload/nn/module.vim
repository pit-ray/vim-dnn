vim9script

import autoload 'autograd.vim' as ag

var Tensor = ag.Tensor


export interface HasModule
  def GetParameters(): list<Tensor>
  def Forward(...inputs: list<any>): Tensor
endinterface
