vim9script

import autoload 'autograd.vim' as ag
import './optimizer.vim'

var HasOptimizer = optimizer.HasOptimizer
var Tensor = ag.Tensor


export class SGD implements HasOptimizer
  this.params: list<Tensor>
  this.lr: float
  this.momentum: float
  this.weight_decay: float
  this.vs: dict<Tensor>

  def new(
      params: list<Tensor>,
      lr: float = 0.01,
      momentum: float = 0.9,
      weight_decay: float = 0.0)
    this.params = params
    this.lr = lr
    this.momentum = momentum
    this.weight_decay = weight_decay
  enddef

  def OneUpdate(param: Tensor): Tensor
    if this.weight_decay > 0
      ag.Elementwise(
        [param.grad, param],
        (g, p): float => g + this.weight_decay * p,
        param.grad)
    endif

    if this.momentum == 0
      return ag.Elementwise(
        [param, param.grad], (p, g): float => p - g * this.lr, param)
    endif

    if !this.vs->has_key(param.id)
      this.vs[param.id] = ag.ZerosLike(param)
    endif

    var v: Tensor = this.vs[param.id]
    v = ag.Sub(ag.Mul(v, this.momentum), ag.Mul(this.lr, param.grad))
    this.vs[param.id] = v

    return ag.Elementwise([param, v], (x1, x2): float => x1 + x2, param)
  enddef

  def Step()
    map(this.params, (_, v): Tensor => this.OneUpdate(v))
  enddef

  def ZeroGrad()
    for param in this.params
      param.grad = ag.ZerosLike(param)
    endfor
  enddef
endclass
