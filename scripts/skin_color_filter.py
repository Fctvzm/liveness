import sys
import numpy

class SkinColorFilter():

	def __init__(self):
		self.mean = numpy.array([0.0, 0.0])
		self.covariance = numpy.zeros((2, 2), 'float64')
		self.covariance_inverse = numpy.zeros((2, 2), 'float64')

	def __generate_circular_mask(self, image, radius_ratio=0.4):
		x_center = image.shape[1] / 2
		y_center = image.shape[2] / 2

		x = numpy.zeros((image.shape[1], image.shape[2]))
		x[:] = range(0, x.shape[1])
		y = numpy.zeros((image.shape[2], image.shape[1]))
		y[:] = range(0, y.shape[1])
		y = numpy.transpose(y)

		x -= x_center
		y -= y_center

		radius = radius_ratio*image.shape[2]
		self.circular_mask = (x**2 + y**2) < (radius**2)


	def __remove_luma(self, image):
		luma = 0.299*image[0, self.circular_mask] + 0.587*image[1, self.circular_mask] + 0.114*image[2, self.circular_mask]
		m = numpy.mean(luma)
		s = numpy.std(luma)

		luma = 0.299*image[0, :, :] + 0.587*image[1, :, :] + 0.114*image[2, :, :]
		self.luma_mask = numpy.logical_and((luma > (m - 1.5*s)), (luma < (m + 1.5*s)))


	def estimate_gaussian_parameters(self, image):
		self.__generate_circular_mask(image)
		self.__remove_luma(image)
		mask = numpy.logical_and(self.luma_mask, self.circular_mask)

		channel_sum = image[0].astype('float64') + image[1] + image[2]
		nonzero_mask = numpy.logical_or(numpy.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
		r = numpy.zeros((image.shape[1], image.shape[2]))
		r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
		g = numpy.zeros((image.shape[1], image.shape[2]))
		g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]
		self.mean = numpy.array([numpy.mean(r[mask]), numpy.mean(g[mask])])

		r_minus_mean = r[mask] - self.mean[0]
		g_minus_mean = g[mask] - self.mean[1]
		samples = numpy.vstack((r_minus_mean, g_minus_mean))
		samples = samples.T
		cov = sum([numpy.outer(s,s) for s in samples])
		self.covariance = cov / float(samples.shape[0] - 1) 

		if numpy.linalg.det(self.covariance) != 0:
			self.covariance_inverse = numpy.linalg.inv(self.covariance)
		else:
			self.covariance_inverse = numpy.zeros_like(self.covariance)


	def get_skin_mask(self, image, threshold):
		skin_map = numpy.zeros((image.shape[1], image.shape[2]), 'float64')

		channel_sum = image[0].astype('float64') + image[1] + image[2]
		nonzero_mask = numpy.logical_or(numpy.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
		r = numpy.zeros((image.shape[1], image.shape[2]), 'float64')
		r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
		g = numpy.zeros((image.shape[1], image.shape[2]), 'float64')
		g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]

		r_minus_mean = r - self.mean[0]
		g_minus_mean = g - self.mean[1]
		v = numpy.dstack((r_minus_mean, g_minus_mean))
		v = v.reshape((r.shape[0]*r.shape[1], 2))
		probs = [numpy.dot(k, numpy.dot(self.covariance_inverse, k)) for k in v]
		probs = numpy.array(probs).reshape(r.shape)
		skin_map = numpy.exp(-0.5 * probs)

		return skin_map > threshold