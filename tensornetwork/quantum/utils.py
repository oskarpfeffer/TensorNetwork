# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
File: tensornetwork/quantum/utils.py
Contact: oskar.pfeffer@ptb.de
Github: https://github.com/oskarpfeffer/TensorNetwork
Description: Utility functions for quantum tensor networks.
"""

import jax
import jax.numpy as jnp
import math


def get_axes_size(tensor: jax.Array, axes_index):
  """Returns the size of the axes of a tensor given an index.

  Args:
      tensor (tensor): tensor of which the axes size will be retrieved
      axes_index (tuple): index of the axes of which to get the size
  Returns:
      tuple: tuple with axes size

  """
  return tuple(tensor.shape[i] for i in axes_index)


def get_positive_index(tensor: jax.Array, axes_index: tuple or int):
  """Returns index % tensor.ndim.

  Args:
      tensor (jax.Array): tensor for which to compute the positive index
      axes_index (tuple or int): index to be made positive
  Returns:
      tuple or int: positive index

  """
  if isinstance(axes_index, int):
    if not -tensor.ndim <= axes_index < tensor.ndim:
      raise ValueError(
        f"axis {axes_index} is out of bounds for array of dimension {tensor.ndim}"
      )
    return axes_index % tensor.ndim
  if any(not -tensor.ndim <= index < tensor.ndim for index in axes_index):
    raise ValueError(
      f"axis tuple {axes_index} is out of bounds for array of dimension {tensor.ndim}"
    )
  if len(set(axes_index)) < len(axes_index):
    raise ValueError(f"axis tuple {axes_index} has repeated axes indices")
  return tuple(index % tensor.ndim for index in axes_index)


def is_unitary(
  tensor: jax.Array,
  in_axes_index: tuple = (-2,),
  out_axes_index: tuple = (-1,),
  tol: float = 1e-10,
):
  """Check whether tensor is unitary along given computational axes.

  The unitarity condition is met, when all the matrices along the computational axis satisfy it.

  Args:
      tensor (jax.Array): tensor to check for unitarity
      in_axes_index (tuple): input axes of the operator
      out_axes_index (tuple): output axes of the operator
      tol (float): numerical tolerance
  Returns:
      bool: True if unitary, False else

  """
  tensor = transpose_join_computational_axes(tensor, in_axes_index, out_axes_index)
  if tensor.shape[-2] != tensor.shape[-2]:
    return False

  Identity = jnp.eye(tensor.shape[-1])
  tensor_dag = jnp.linalg.matrix_transpose(tensor.conj())

  left = jnp.matmul(tensor_dag, tensor)
  right = jnp.matmul(tensor, tensor_dag)

  return jnp.allclose(left, Identity, atol=tol) and jnp.allclose(
    right, Identity, atol=tol
  )


def join_computational_axes(
  tensor: jax.Array, in_axes_size: tuple, out_axes_size: tuple, invert: bool = False
):
  """Join the computational axes when they are in the end.

  Reshapes a tensor of shape [..., in_axes_size, out_axes_size] to a tensor of shape [..., math.prod(in_axes_size), math.prod(out_axes_size)].
  If invert, it does the inverse transformation.

  Args:
      tensor: tensor to be transformed
      in_axes_size: list with input axes size
      out_axes_size: list with output axes size
      invert: if True performs backward transformation
  Returns:
      Reshaped tensor

  """
  old_shape = tensor.shape
  if not invert:
    new_shape = old_shape[: -len(in_axes_size) - len(out_axes_size)]
    new_shape += (math.prod(in_axes_size),) + (math.prod(out_axes_size),)
  else:
    new_shape = old_shape[:-2]
    new_shape += in_axes_size + out_axes_size
  tensor = tensor.reshape(new_shape)
  return tensor


def transpose_computational_axes(
  tensor, in_axes_index: tuple, out_axes_index: tuple, invert: bool = False
):
  """Transposes an operator to have `in_axes_index + out_axes_index` as last
  axes.

  Args:
      tensor (jax.Array): tensor to be transposed
      in_axes_index (tuple): input axes of the operator
      out_axes_index (tuple): output axes of the operator
      invert (bool): if True performs backward transformation. Defaults to False.
  Returns:
      transposed tensor

  """
  in_axes_index = get_positive_index(tensor, in_axes_index)
  out_axes_index = get_positive_index(tensor, out_axes_index)
  transpose_index = [
    axis for axis in range(tensor.ndim) if axis not in in_axes_index + out_axes_index
  ]
  transpose_index += in_axes_index + out_axes_index
  if invert:
    transpose_index = [transpose_index.index(i) for i in range(tensor.ndim)]
  tensor = tensor.transpose(transpose_index)
  return tensor


def transpose_join_computational_axes(
  tensor,
  in_axes_index: tuple,
  out_axes_index: tuple,
  in_axes_size: tuple = None,
  out_axes_size: tuple = None,
  invert: bool = False,
):
  """Transposes and joins the computational axes as last axes.

  Args:
      tensor (jax.Array): tensor to be transformed
      in_axes_index (tuple): input_axes of the operator
      out_axes_index (tuple): output axes of the operator
      in_axes_size (tuple, optional): list with input axes size. Needed when invert. Defaults to None.
      out_axes_size (tuple, optional): list with output axes size. Needed when invert. Defaults to None.
      invert (bool, optional): if True performs backward transformation. Defaults to False.

  Raises:
      ValueError: If invert, then in_axes_size and out_axes_size are needed

  Returns:
      jax.Array: transformed tensor

  """
  if not invert:
    in_axes_size = get_axes_size(tensor, in_axes_index)
    out_axes_size = get_axes_size(tensor, out_axes_index)
    tensor = transpose_computational_axes(tensor, in_axes_index, out_axes_index)
    tensor = join_computational_axes(tensor, in_axes_size, out_axes_size)
  else:
    if in_axes_size is None or out_axes_size is None:
      raise ValueError(
        "When invert, in_axes_size and out_axes_size must be passed as arguments"
      )
    tensor = join_computational_axes(tensor, in_axes_size, out_axes_size, invert)
    tensor = transpose_computational_axes(tensor, in_axes_index, out_axes_index, invert)
  return tensor
