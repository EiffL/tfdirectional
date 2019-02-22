# Copyright 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The Bingham distribution over vectors on the unit hypersphere."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization


__all__ = ['Bingham']

class Bingham(distribution.Distribution):
    r""" The Bingham distribution over unit vectors on `S^{n-1}`
    TODO: refine doc
    Implementation based on: https://github.com/libDirectional/libDirectional/blob/master/lib/distributions/Hypersphere/BinghamDistribution.m
    """

    def __init__(self,
                 rotation,
                 concentration,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Bingham'):
        """Creates a new  `Bingham` instance.

        Args:
            rotation: Floating-point `Tensor` of shape [B1, ... Bn, D, D]
                Orthogonal matrix that describes rotation

            concentration: Floating-point `Tensor` having batch shape [B1, ... Bn, D]
                Concentration parameters, have to be increasing, and last entry is 0
        """
        parameters = dict(locals())
        with tf.name_scope(name, values=[rotation, concentration]) as name:
            dtype = dtype_util.common_dtype([rotation, concentration],
                                                    tf.float32)
            rotation = tf.convert_to_tensor(
                                value=rotation, name='rotation', dtype=dtype)
            concentration = tf.convert_to_tensor(
                        value=concentration, name='concentration', dtype=dtype)

            # Checks should  verify that rotation is orthogonal, last c is 0, ext
            assertions = []
            with tf.control_dependencies(assertions):
                self._rotation = tf.identity(rotation)
                self._concentration = tf.identity(concentration)

            # For now, not reparameterized
            reparameterization_type = reparameterization.NOT_REPARAMETERIZED

            super(self.__class__, self).__init__(dtype=self._concentration.dtype,
                        validate_args=validate_args,
                        allow_nan_stats=allow_nan_stats,
                        reparameterization_type=reparameterization_type,
                        parameters=parameters,
                        graph_parents=[self._rotation, self._concentration],
                        name=name)

    @property
    def rotation(self):
        """ Rotation parameter """
        return self._rotation

    @property
    def concentration(self):
        """ Concentration parameters """
        return self._concentration

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.shape(input=self.rotation)[:-2],
            tf.shape(input=self.concentration[:-1]))

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self.rotation.shape.with_rank_at_least(1)[:-2],
            self.concentration.shape[:-1])

    def _event_shape_tensor(self):
        return tf.shape(input=self.concentration)[-1:]

    def _log_prob(self, x):
        x = self._maybe_assert_valid_sample(x)
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, samples):
        samples = self._maybe_assert_valid_sample(samples)

        C = tf.matmul(self.rotation, tf.matmul( tf.linalg.diag(self.concentration), self.rotation, transpose_b=True))

        inner_product = tf.reduce_sum(
            input_tensor=samples * tf.matmul(C, tf.expand_dims(samples,axis=-1))[...,0], axis=-1)
        return inner_product

    def _log_normalization(self):
        """Computes the log-normalizer of the distribution."""
        # This is a formula that only works for  the d=2
        s2 = 1 # should be BinghamDistribution.S2
        F = tf.exp(self.concentration[...,-1]) * s2 * tf.math.bessel_i0e((self.concentration[...,0] - self.concentration[...,1])/2) * tf.exp((self.concentration[...,0] - self.concentration[...,1]));
        return F

    def _maybe_assert_valid_sample(self, samples):
        """Check counts for proper shape, values, then return tensor version."""
        if not self.validate_args:
            return samples
        with tf.control_dependencies([
            tf.compat.v1.assert_near(
                1.,
                tf.linalg.norm(tensor=samples, axis=-1),
                message='samples must be unit length'),
            tf.compat.v1.assert_equal(
                tf.shape(input=samples)[-1:],
                self.event_shape_tensor(),
                message=('samples must have innermost dimension matching that of '
                         '`self.concentration`')),
            ]):
            return tf.identity(samples)
