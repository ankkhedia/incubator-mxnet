# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""Contrib datasets."""

# from ...gluon import EventHandler

from .EventHandler import LoggingHandler, CheckpointHandler, MetricHandler

from ... import *
from ... import gluon, autograd
import time

__all__ = ['Estimator']

class Estimator:
    def __init__(self, net, metric = [metric.RMSE()], loss= gluon.loss.SoftmaxCrossEntropyLoss(), evaluate_every=5):
        self._train_stats = {"lr": [], "epoch": []}
        self.CheckpointHandler = CheckpointHandler(self)
        self.LoggingHandler = LoggingHandler(self)
        self.MetricHandler= MetricHandler(self)
        self._net = net
        self._epoch = 0
        self._metric= metric
        self._lossfn = loss
        self._evaluate_every= evaluate_every
        self.y_hat= None
        self.y= None
        self.X=None

    def plotting_fn(self):
        pass

    # def evaluate_accuracy(data_iter, net, ctx):
    #     acc_sum, n = nd.array([0], ctx=ctx), 0
    #     for X, y in data_iter:
    #         # If ctx is the GPU, copy the data to the GPU.
    #         X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
    #         acc_sum += (net(X).argmax(axis=1) == y).sum()
    #         n += y.size
    #   return acc_sum.asscalar() / n

    def _batch_fn(self,batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    #def evaluate_loss_and_metric(self, dataloader):
    #    for (i,batch) in enumerate(dataloader):
    #       X= batch.data
    #        y= batch.label
    #        y_pred = self._net(X)
    #        ##replace this with custom function just as discussed
    #        ## Also update val_loss thing
    #        self._metric.update(y_pred,y)
    #    metric_eval= self._metric.get_name_value()
    #    for name, val in metric_eval:
    #        self._train_stats['val_'+str(name)].append(val)
    #    return metric_eval


    def fit(self, train_data_loader, val_data_loader, epochs,
            trainers= None,
            ctx=[Context.default_ctx], additionalHandlers=[], batch_size=64):
        self._net.initialize(force_reinit=True, ctx = ctx, init=init.Xavier())
        #self._metric = train_metric
        if trainers is None:
            trainers = gluon.Trainer(self._net.collect_params(), 'sgd', {'learning_rate': 0.001})
        EventHandlers = [self.LoggingHandler, self.CheckpointHandler, self.MetricHandler]
        self.MetricHandler._valdataloader= val_data_loader
        EventHandlers = EventHandlers + additionalHandlers
        exit_training = False
        #print(self._metric.get())
        #m_name, m_val =
        #for metrics in self._metric:
        #    train_metric_name, train_metric_val = zip(*(metrics.get_name_value()))
        #    for m_names in train_metric_name:
        #        #print(self._metric.get()[0])
        #        #print(m_names)
        #        self._train_stats[m_names]=[]
        for handlers in EventHandlers:
            handlers.train_begin()


        for epoch in range(epochs):
            print(epoch)
            for handlers in EventHandlers:
                handlers.epoch_begin()
            for metrics in self._metric:
                metrics.reset()
            if exit_training:
                break
            for i, batch in enumerate(train_data_loader):
                #data, label = self._batch_fn(batch, ctx)
                print("inside batch")
                for handlers in EventHandlers:
                    handlers.batch_begin()
                #train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
                #for X, y in train_data_loader:
                    #X, y = X.as_in_context(ctx), y.as_in_context(ctx)
                    #sum =0
                self.X= batch.data
                self.y=batch.label
                with autograd.record():
                    y_hat = self._net(X)
                    l= self._lossfn(y_hat, self.y, weight= None)

                l.backward()
                trainers.step(batch_size)
                y = y.astype('float32')
                #train_l_sum += l.asscalar()
                #train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
                #n += y.size
                for handlers in EventHandlers:
                    handlers.batch_end()


                print("end of batch")
                break
            #metric_val = metrics.get_name_value()
            #print(train_metric_name)
            #print(train_metric_score)

            self._train_stats["epoch"].append(epoch)
            self._train_stats["lr"].append(trainers.learning_rate)
            #for name, val in metric_val:
            #    self._train_stats[name].append(val)
            print(self._train_stats)
            ## how to do this for val

            #print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '

            #      'time %.1f sec'
            #      % (epoch + 1, train_l_sum / n, train_acc_sum / n, train_metric_score,
            #         time.time() - start))

            #if self._epoch % self._evaluate_every == 0:
            #    self.evaluate_loss_and_metric(val_data_loader)
            for handlers in EventHandlers:
                print(handlers)
                handlers.epoch_end()
            self._epoch = epoch +1
            print(self._train_stats)
        print ("completed epochs")
        self.plotting_fn()
        print("plots")
        for handlers in EventHandlers:
            handlers.train_end()
        print("end of fit")
    # def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    #        pass

