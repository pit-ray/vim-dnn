# vim-dnn

## Dependencies

- [vim-autograd](https://github.com/pit-ray/vim-autograd)

```vim
Plug 'pit-ray/vim-autograd', {'branch': 'vim9'}
```


## Usage

```vim

import autoload 'autograd.vim' as ag
import autoload 'dnn.vim' as nn

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
```
