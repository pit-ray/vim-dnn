vim9script

import autoload 'autograd.vim' as ag

var Tensor = ag.Tensor


export def ReLU(x: any): Tensor
  return ag.Maximum(x, 0.0)
enddef


export def Softmax(x: any): Tensor
  var y: Tensor = ag.Exp(ag.Sub(x, ag.Max(x)))
  var sx = ag.Sum(y, 1, true)
  return ag.Div(y, sx)
enddef


export def CrossEntropyLoss(y: any, t: any): Tensor
  var loss: Tensor = ag.Mul(t, ag.Log(y))
  var batch_size = loss.shape[0]
  return ag.Div(ag.Sum(loss), -1 * batch_size)
enddef
