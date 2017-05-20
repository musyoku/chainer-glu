# encoding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import numpy as np
from qrnn import QRNN, QRNNEncoder, QRNNDecoder, QRNNGlobalAttentiveDecoder
from convolution_1d import Convolution1D

def test_convolution_1d():
	ksize = 4
	layer = Convolution1D(3, 30, ksize, pad=ksize-1)
	x_shape = (2, 3, 5)
	x = np.random.uniform(-1, 1, x_shape)
	print(layer(x))

def test_decoder():
	np.random.seed(0)
	enc_shape = (2, 3, 5)
	dec_shape = (2, 4, 7)
	prod = enc_shape[0] * enc_shape[1] * enc_shape[2]
	enc_data = np.arange(0, prod, dtype=np.float32).reshape(enc_shape) / prod
	prod = dec_shape[0] * dec_shape[1] * dec_shape[2]
	dec_data = np.arange(0, prod, dtype=np.float32).reshape(dec_shape) / prod
	skip_mask = np.ones((enc_data.shape[0], enc_data.shape[2]), dtype=np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0

	encoder = QRNNEncoder(enc_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)
	decoder = QRNNDecoder(dec_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)

	np.random.seed(0)
	H = encoder(enc_data, skip_mask)
	ht = encoder.get_last_hidden_state()
	Y = decoder(dec_data, ht)

	np.random.seed(0)
	decoder.reset_state()
	for t in xrange(dec_shape[2]):
		y = decoder.forward_one_step(dec_data[:, :, :t+1], ht)
		assert np.sum((y.data - Y.data[:, :, :t+1]) ** 2) == 0
		print("t = {} OK".format(t))


def test_attentive_decoder():
	np.random.seed(0)
	enc_shape = (2, 3, 5)
	dec_shape = (2, 4, 7)
	prod = enc_shape[0] * enc_shape[1] * enc_shape[2]
	enc_data = np.arange(0, prod, dtype=np.float32).reshape(enc_shape) / prod
	prod = dec_shape[0] * dec_shape[1] * dec_shape[2]
	dec_data = np.arange(0, prod, dtype=np.float32).reshape(dec_shape) / prod
	skip_mask = np.ones((enc_data.shape[0], enc_data.shape[2]), dtype=np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0

	encoder = QRNNEncoder(enc_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)
	decoder = QRNNGlobalAttentiveDecoder(dec_shape[1], 4, kernel_size=4, zoneout=False, zoneout_ratio=0.5)

	H = encoder(enc_data, skip_mask)
	ht = encoder.get_last_hidden_state()
	Y = decoder(dec_data, ht, H, skip_mask)

	decoder.reset_state()
	for t in xrange(dec_shape[2]):
		y = decoder.forward_one_step(dec_data[:, :, :t+1], ht, H, skip_mask)
		assert np.sum((y.data - Y.data[:, :, :t+1]) ** 2) == 0
		print("t = {} OK".format(t))


import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import initializers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from chainer.utils import conv_nd
import convolution_1d


@testing.parameterize(*(testing.product({
	'dims': [(5,)],
	'dtype': [numpy.float32]
}) + testing.product({
	'dims': [(5,)],
	'dtype': [numpy.float16, numpy.float32, numpy.float64]
})))
class TestConvolutionND(unittest.TestCase):

	def setUp(self):
		self.ksize = (3,)
		self.stride = (2,)
		self.pad = (1,)

		self.link = convolution_1d.Convolution1D(3, 2, self.ksize, initialV=initializers.Normal(scale=1, dtype=self.dtype), stride=self.stride, pad=self.pad)
		x_shape = (2, 3) + self.dims
		self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
		self.link(self.x)	# initialize g and b

		self.link.cleargrads()
		gy_shape = (2, 2) + tuple(
			conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(
				self.dims, self.ksize, self.stride, self.pad))
		self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

		self.check_backward_options = {'eps': 1e-2, 'atol': 1e-3, 'rtol': 1e-3}
		if self.dtype == numpy.float16:
			self.check_backward_options = {
				'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4}

	@attr.gpu
	def test_im2col_consistency(self):
		col_cpu = conv_nd.im2col_nd_cpu(
			self.x, self.ksize, self.stride, self.pad)
		col_gpu = conv_nd.im2col_nd_gpu(
			cuda.to_gpu(self.x), self.ksize, self.stride, self.pad)
		testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

	@attr.gpu
	def test_col2im_consistency(self):
		col = conv_nd.im2col_nd_cpu(self.x, self.ksize, self.stride, self.pad)
		im_cpu = conv_nd.col2im_nd_cpu(col, self.stride, self.pad, self.dims)
		im_gpu = conv_nd.col2im_nd_gpu(
			cuda.to_gpu(col), self.stride, self.pad, self.dims)
		testing.assert_allclose(im_cpu, im_gpu.get())

	def check_forward_consistency(self):
		x_cpu = chainer.Variable(self.x)
		y_cpu = self.link(x_cpu)
		self.assertEqual(y_cpu.data.dtype, self.dtype)

		self.link.to_gpu()
		x_gpu = chainer.Variable(cuda.to_gpu(self.x))
		y_gpu = self.link(x_gpu)
		self.assertEqual(y_gpu.data.dtype, self.dtype)

		testing.assert_allclose(y_cpu.data, y_gpu.data.get())

	@attr.cudnn
	@condition.retry(3)
	def test_forward_consistency(self):
		self.check_forward_consistency()

	@attr.gpu
	@condition.retry(3)
	def test_forward_consistency_im2col(self):
		self.link.use_cudnn = False
		self.check_forward_consistency()

	def check_backward(self, x_data, y_grad):
		gradient_check.check_backward(self.link, x_data, y_grad, (self.link.V, self.link.g, self.link.b), **self.check_backward_options)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.x, self.gy)

	@attr.cudnn
	@condition.retry(3)
	def test_backward_gpu(self):
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	@attr.gpu
	@condition.retry(3)
	def test_backward_gpu_im2col(self):
		self.link.use_cudnn = False
		self.link.to_gpu()
		self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

	def check_pickling(self, x_data):
		x = chainer.Variable(x_data)
		y = self.link(x)
		y_data1 = y.data

		del x, y

		pickled = pickle.dumps(self.link, -1)
		del self.link
		self.link = pickle.loads(pickled)

		x = chainer.Variable(x_data)
		y = self.link(x)
		y_data2 = y.data

		testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

	def test_pickling_cpu(self):
		self.check_pickling(self.x)

	@attr.gpu
	def test_pickling_gpu(self):
		self.link.to_gpu()
		self.check_pickling(cuda.to_gpu(self.x))


class TestConvolutionNDNoInitialBias(unittest.TestCase):

	def test_no_initial_bias(self):
		ksize = 3
		link = convolution_1d.Convolution1D(3, 2, ksize, nobias=True)
		self.assertIsNone(link.b)


if __name__ == "__main__":
	test_convolution_1d()
	test_decoder()
	test_attentive_decoder()
	testing.run_module(__name__, __file__)