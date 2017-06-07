from __future__ import division
from __future__ import print_function
from six.moves import xrange
import math
import numpy as np
import chainer
from chainer import cuda, Variable, function, link, functions, links, initializers
from chainer.utils import type_check
from chainer.links import EmbedID, Linear, BatchNormalization, ConvolutionND
from convolution_1d import Convolution1D as WeightnormConvolution1D

def Convolution1D(in_channels, out_channels, ksize, stride=1, pad=0, initialW=None, weightnorm=False):
	if weightnorm:
		return WeightnormConvolution1D(in_channels, out_channels, ksize, stride=stride, pad=pad, initialV=initialW)
	return ConvolutionND(1, in_channels, out_channels, ksize, stride=stride, pad=pad, initialW=initialW)

class GLU(link.Chain):
	def __init__(self, in_channels, out_channels, kernel_size=2, wgain=1., weightnorm=False):
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		super(GLU, self).__init__(W=Convolution1D(in_channels, 2 * out_channels, kernel_size, stride=1, pad=kernel_size - 1, weightnorm=weightnorm, initialW=initializers.HeNormal(wstd)))
		self._in_channels, self._out_channels, self._kernel_size, = in_channels, out_channels, kernel_size
		self.reset_state()

	def __call__(self, X):
		# remove right paddings
		# e.g.
		# kernel_size = 3
		# pad = 2
		# input sequence with paddings:
		# [0, 0, x1, x2, x3, 0, 0]
		# |< t1 >|
		#     |< t2 >|
		#         |< t3 >|
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, :-pad]

		A, B = functions.split_axis(WX, 2, axis=1)
		self.H = A * functions.sigmoid(B)
		return self.H

	def forward_one_step(self, X):
		pad = self._kernel_size - 1
		wx = self.W(X)[:, :, -pad-1, None]
		a, b = functions.split_axis(wx, 2, axis=1)
		h = a * functions.sigmoid(b)

		if self.H is None:
			self.H = h
		else:
			self.H = functions.concat((self.H, h), axis=2)

		return self.H

	def reset_state(self):
		self.set_state(None)

	def set_state(self, H):
		self.H = H

	def get_all_hidden_states(self):
		return self.H
