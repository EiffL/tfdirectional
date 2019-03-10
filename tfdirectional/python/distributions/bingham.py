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
import math

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution, MultivariateNormalFullCovariance
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

        Only implemented for D =3 for now

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

            # Checks should verify that rotation is orthogonal, last c is 0, ext
            assertions = []
            with tf.control_dependencies(assertions):
                self._rotation = tf.identity(rotation)
                self._concentration = tf.identity(concentration)

            # Compute sphere surface area
            # TODO: generalize to different dimensions
            self._s2 = 2. * math.pi**((3.)/2)/math.gamma((3.)/2);

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
            tf.shape(input=self.concentration)[:-1])

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
        # This is a formula that only works for  the d=2, which is useless for us :-/
        #F = tf.exp(self.concentration[...,-1]) * self._s2 * tf.math.bessel_i0e((self.concentration[...,0] - self.concentration[...,1])/2) * tf.exp((self.concentration[...,0] - self.concentration[...,1]));

        # Compute normalization constant using the saddle point method
        # Define the derivatives of the cumulant generative function
        def K1(t):
             return tf.reduce_sum(0.5 / (tf.sqrt(-self.concentration) - t), axis=-1)
        def K2(t):
             return tf.reduce_sum(0.5 / (tf.sqrt(-self.concentration) - t)**2, axis=-1)
        def K3(t):
             return tf.reduce_sum(1. / (tf.sqrt(-self.concentration) - t)**3, axis=-1)
        def K4(t):
             return tf.reduce_sum(3. / (tf.sqrt(-self.concentration) - t)**4, axis=-1)

        # Step 1: Finding the root tstar
        def f(x):
            fx = K1(x) - 1.
            dx = tf.gradients(fx, x)
            return tf.reshape(fx / dx, [-1])

        def newton_raphson_body(x, x_0, p, k):
            x_0 = tf.identity(x)
            x = x_0 - f(x_0)
            k = tf.add(k, 1)
            return [x, x_0, p, k]

        def newton_raphson_condition(x, x_0, p, k):
            return tf.reduce_max(tf.abs(x - x_0)) > p

        x0 = tf.ones(batch_size, dtype=self.dtype) * minEl-0.5
        x  = x0 - f(x0)
        p  = tf.constant(name="precision", shape=[], dtype=self.dtype, value=precision)
        k  = tf.zeros(shape=[], dtype=tf.int64)
        loop = tf.while_loop(newton_raphson_condition,
                             newton_raphson_body, loop_vars=[x, x0, p, k])



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

    def _sample_n(self, n, seed=None, precision=1e-6):
        """ Sampling procedure based on Kent's algorithm
        """
        seed = seed_stream.SeedStream(seed, salt='bingham')
        event_dim = ( self.event_shape[0].value or
            self._event_shape_tensor()[0])

        batch_size = self._batch_shape_tensor()
        sample_batch_shape = tf.concat([[n]], axis=0)
        dim = tf.cast(event_dim, self.dtype)

        q = dim
        A = - tf.matmul(self.rotation, tf.matmul( tf.linalg.diag(self.concentration), self.rotation, transpose_b=True))

        # Implement root finding algorithm to compute b
        def f(x):
            fx = tf.reduce_sum( 1./(tf.expand_dims(x,axis=-1) - 2.*self.concentration) ,axis=-1) - 1.
            dx = tf.gradients(fx, x)
            return tf.reshape(fx / dx, [-1])

        def newton_raphson_body(x, x_0, p, k):
            x_0 = tf.identity(x)
            x = x_0 - f(x_0)
            k = tf.add(k, 1)
            return [x, x_0, p, k]

        def newton_raphson_condition(x, x_0, p, k):
            return tf.reduce_max(tf.abs(x - x_0)) > p

        x0 = tf.ones(batch_size, dtype=self.dtype)
        x  = x0 - f(x0)
        p  = tf.constant(name="precision", shape=[], dtype=self.dtype, value=precision)
        k  = tf.zeros(shape=[], dtype=tf.int64)
        loop = tf.while_loop(newton_raphson_condition,
                             newton_raphson_body, loop_vars=[x, x0, p, k])
        # Extract the root
        b = loop[0]
        Omega = tf.eye(3) + 2*A/tf.expand_dims(tf.expand_dims(b,axis=-1),axis=-1)
        Mbstar = tf.exp(-(q - b)/2) *(q / b)**(q / 2.)

        # Build angular central Gaussian
        mvnrnd = MultivariateNormalFullCovariance(loc=tf.zeros(3, dtype=self.dtype),
                                                  covariance_matrix=tf.linalg.inv(Omega))

        w = tf.zeros(batch_size*n*event_dim, dtype=self.dtype)
        w = tf.reshape(w, [-1, event_dim])
        should_continue = tf.ones(n * batch_size, dtype=tf.bool)

        # Sampling loop
        def cond_fn(w, should_continue):
            del w
            return tf.reduce_any(input_tensor=should_continue)

        def body_fn(w, should_continue):
            # Draw from angular central Gaussian
            y = mvnrnd.sample(sample_shape=sample_batch_shape, seed=seed())
            x = tf.nn.l2_normalize(y, axis=-1)

            # Update the samples where needed
            w = tf.where(should_continue, tf.reshape(x, [-1, event_dim]), w)
            w = tf.debugging.check_numerics(w, 'w')

            y =  tf.transpose(x, [1,2,0])
            a = tf.transpose(tf.reduce_sum(y*tf.matmul(A,y),axis=1), [1,0])
            b = tf.transpose(tf.reduce_sum(y*tf.matmul(Omega,y),axis=1), [1,0])
            c = tf.random.uniform(batch_size*sample_batch_shape, seed=seed(), dtype=self.dtype) <  tf.reshape(tf.exp(-a) / (Mbstar * b**(q/2.)), [-1])
            should_continue = tf.logical_and(should_continue, c)
            return w, should_continue


        samples = tf.while_loop(cond=cond_fn, body=body_fn, loop_vars=(w, should_continue))[0]

        samples = tf.reshape(samples, [n, -1, event_dim])
        return samples
