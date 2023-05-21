vim9script

import autoload 'autograd.vim' as ag
import '../autoload/dnn.vim' as nn

var Tensor = ag.Tensor
var HasModule = nn.HasModule


class MLP implements HasModule
  this.l1: HasModule
  this.l2: HasModule

  def new(in_size: number, classes: number, hidden_size: number = 100)
    this.l1 = nn.Linear.new(in_size, hidden_size)
    this.l2 = nn.Linear.new(hidden_size, classes)
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


def GetWineDataset(filename: string): list<list<list<float>>>
  # This refers to the following public toy dataset.
  # http//archive.ics.uci.edu/ml/datasets/Wine
  var dataset = map(
    readfile(filename),
    (_, line) => map(
      split(line, ','),
      (_, v) => str2float(v)))

  var N = len(dataset)

  # average
  var means = repeat([0.0], 14)
  for data in dataset
    for i in range(1, 13)
      means[i] += data[i]
    endfor
  endfor
  map(means, (_, v) => v / N)

  # standard deviation
  var stds = repeat([0.0], 14)
  for data in dataset
    for i in range(1, 13)
      stds[i] += pow(data[i] - means[i], 2)
    endfor
  endfor
  map(stds, (_, v) => sqrt(v / N))

  # standardization
  for data in dataset
    for i in range(1, 13)
      data[i] = (data[i] - means[i]) / stds[i]
    endfor
  endfor

  # split the dataset into train and test.
  var train_x: list<list<float>>
  var train_t: list<list<float>>
  var test_x: list<list<float>>
  var test_t: list<list<float>>
  var test_num_per_class = 10
  for i in range(3)
    var class_split = ag.Shuffle(
      filter(deepcopy(dataset), (_, v) => v[0] == i + 1))

    var train_split = class_split[: -test_num_per_class - 1]
    var test_split = class_split[-test_num_per_class :]

    train_x += mapnew(train_split, (_, vs) => vs[1 :])
    train_t += mapnew(train_split, (_, vs) => map(vs[: 0], (_, v) => v - 1))
    test_x += mapnew(test_split, (_, vs) => vs[1 :])
    test_t += mapnew(test_split, (_, vs) => map(vs[: 0], (_, v) => v - 1))
  endfor

  return [train_x, train_t, test_x, test_t]
enddef


def Main()
  ag.ManualSeed(42)

  var [train_x, train_t, test_x, test_t] = GetWineDataset('.dnn/wine.data')
  var ndim = 13
  var nclass = 3
  var model = MLP.new(ndim, nclass)
  var optim = nn.SGD.new(model.GetParameters(), 0.1, 0.9, 0.0001)

  # train
  var max_epoch: number = 14
  var batch_size: number = 16
  var train_data_num: number = len(train_x)
  var each_iteration: number = float2nr(ceil(1.0 * train_data_num / batch_size))

  var logs: list<string>
  for epoch in range(max_epoch)
    var indices = ag.Shuffle(range(train_data_num))
    var epoch_loss: float = 0.0
    for i in range(each_iteration)
      var x: list<list<float>>
      var t: list<list<float>>
      for index in indices[i * batch_size : (i + 1) * batch_size - 1]
        x->add(train_x[index])

        var onehot = repeat([0.0], nclass)
        onehot[float2nr(train_t[index][0])] = 1.0
        t->add(onehot)
      endfor

      var y = model.Forward(x)
      var loss = nn.CrossEntropyLoss(y, t)
      # ag.DumpGraph(loss, '.dnn/loss.png')

      optim.ZeroGrad()
      ag.Backward(loss)
      optim.Step()

      epoch_loss += loss.data[0]
    endfor

    epoch_loss /= each_iteration

    # logging
    logs->add(epoch .. ', ' .. epoch_loss)
    logs->writefile('.dnn/train.log')
  endfor

  var Argmax = (pred: Tensor): number => {
    var pred_max: Tensor = ag.Max(pred)
    return pred.data->index(pred_max.data[0])
  }

  # evaluate
  ag.NoGrad(() => {
    var accuracy: float = 0.0

    for i in range(len(test_x))
      var pred = model.Forward([test_x[i]])

      if Argmax(pred) == float2nr(test_t[i][0])
        accuracy += 1.0
      endif
    endfor

    echomsg 'accuracy: ' .. accuracy / len(test_t)
  })

enddef


Main()
