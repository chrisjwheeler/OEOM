#from curses import A_BOLD
from typing import Callable
from dataclasses import dataclass, field
import jax
from jax import numpy as jnp, Array

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType


@dataclass
class AmbientDiffusionLoss:
    """
    Loss function for training diffusion models with ambient score matching.

    This dataclass implements various loss functions for diffusion models based on the specified
    target type. The loss is computed as the mean squared error between the model's prediction
    and the target, which depends on the chosen vector field type.

    The loss supports only one target type:
    - ``VectorFieldType.X0``: Learn to predict the original clean data x_0

    Attributes:
        diffusion_process (``DiffusionProcess``): The diffusion process defining the forward dynamics
        num_noise_draws_per_sample (``int``): The number of noise draws per sample to use for the batchwise loss.
        t_n (``float``): The time we denoise to.
    """

    diffusion_process: DiffusionProcess
    num_noise_draws_per_sample: int
    t_n: Array
    target: Callable[[Array, Array, Array, Array, Array], Array] = field(init=False)

    def __post_init__(self):
        pass

    def prediction_loss(
        self, x_t: Array, f_x_t: Array, x_t_n: Array, eps: Array, t: Array
    ) -> Array:
        """
        Compute the loss given a prediction and inputs/targets.

        This method calculates the mean squared error between the model's prediction (``f_x_t``)
        and the target value determined by the target_type (``self.target``).

        Args:
            x_t (``Array[*data_dims]``): The noised data at time ``t``.
            f_x_t (``Array[*data_dims]``): The model's prediction for x0 at time ``t``.
            x_t_n (``Array[*data_dims]``): The noised data at time ``t_n``.
            eps (``Array[*data_dims]``): The noise used to generate ``x_t``.
            t (``Array[]``): The scalar time parameter.

        Returns:
            ``Array[]``: The scalar loss value for the given sample.
        """

        # Parameters
        sigma_t = self.diffusion_process.sigma(t)
        sigma_t_n = self.diffusion_process.sigma(self.t_n)
        one_minus_sqrt_2_t = self.diffusion_process.alpha(t)
        one_minus_sqrt_2_t_n = self.diffusion_process.alpha(self.t_n)

        # Noise Scalings
        model_noise_scaling = (sigma_t**2 - sigma_t_n**2) / (sigma_t**2 * one_minus_sqrt_2_t_n)
        x_t_scaling = (sigma_t_n**2 * one_minus_sqrt_2_t) / (sigma_t**2 * one_minus_sqrt_2_t_n) 

        squared_residuals = ((model_noise_scaling * f_x_t) + (x_t_scaling * x_t - x_t_n)) ** 2
        samplewise_loss = jnp.sum(squared_residuals)
        return samplewise_loss

    def loss(
        self,
        key: Array,
        vector_field: Callable[[Array, Array], Array],
        x_t_n: Array,
        t: Array,
    ) -> Array:
        """
        Compute the average loss over multiple noise draws for a single data point and time.

        This method estimates the expected loss at a given time ``t`` for a clean data sample ``x_0``.
        It does this by drawing ``num_noise_draws_per_sample`` noise vectors (``eps``), generating
        the corresponding noisy samples ``x_t`` using the ``diffusion_process``, predicting the
        target quantity ``f_x_t`` using the provided ``vector_field`` (vmapped internally), and then calculating the
        ``prediction_loss`` for each noise sample. The final loss is the average over these samples.

        Args:
            key (``Array``): The PRNG key for noise generation.
            vector_field (``Callable[[Array, Array], Array]``): The vector field function that takes
                a single noisy data sample ``x_t`` and its corresponding time ``t``, and returns the model's prediction ``f_x_t``.
                This function will be vmapped internally over the batch dimension created by ``num_noise_draws_per_sample``.

                Signature: ``(x_t: Array[*data_dims], t: Array[]) -> Array[*data_dims]``.

            x_0 (``Array[*data_dims]``): The original clean data sample.
            t (``Array[]``): The scalar time parameter.

        Returns:
            ``Array[]``: The scalar loss value, averaged over ``num_noise_draws_per_sample`` noise instances.
        """
        x_t_n_batch = x_t_n[None, ...].repeat(self.num_noise_draws_per_sample, axis=0)
        t_batch = t[None].repeat(self.num_noise_draws_per_sample, axis=0)
        eps_batch = jax.random.normal(key, x_t_n_batch.shape)

        # Compute x_t_s
        
        def ambient_forward_process(x_t_n, t, eps):
            sigma_t = self.diffusion_process.sigma(t)
            sigma_t_n = self.diffusion_process.sigma(self.t_n)
            one_minus_sqrt_2_t = self.diffusion_process.alpha(t)
            one_minus_sqrt_2_t_n = self.diffusion_process.alpha(self.t_n)

            x_t_n_scaling = (one_minus_sqrt_2_t / one_minus_sqrt_2_t_n)
            eps_scaling = (sigma_t**2 - sigma_t_n**2)**(1/2) / one_minus_sqrt_2_t_n

            return (x_t_n_scaling * x_t_n) + (eps_scaling * eps)
        
        batch_diffusion_forward = jax.vmap(
            ambient_forward_process, in_axes=(0, 0, 0)
        )

        x_t_batch = batch_diffusion_forward(x_t_n_batch, t_batch, eps_batch)

        batch_vector_field = jax.vmap(vector_field, in_axes=(0, 0))
        f_x_t_batch = batch_vector_field(x_t_batch, t_batch)

        batch_prediction_loss = jax.vmap(self.prediction_loss, in_axes=(0, 0, 0, 0, 0))
        losses = batch_prediction_loss(
            x_t_batch, f_x_t_batch, x_t_n_batch, eps_batch, t_batch
        )

        loss_value = jnp.mean(losses)
        return loss_value