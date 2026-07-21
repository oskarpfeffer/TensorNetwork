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
File: tensornetwork/base_elements/nodes.py
Contact: oskar.pfeffer@ptb.de
Github: https://github.com/oskarpfeffer/TensorNetwork
Description: Basic nodes to construct tensor networks.
"""

import jax.numpy as jnp
import tensornetwork as tn
import jax
from collections.abc import Callable


def zero_state(dimension: int, name: str = None) -> tn.Node:
  """Return a [1, 0, 0, ...] vector node.

  Args:
      dimension (int): Leg dimension

  Returns:
      tn.Node: zero state node

  """
  if name is None:
    name = "zero state"
  tensor = jnp.zeros(dimension).at[0].set(1)
  return tn.Node(tensor, name=name)


def delta(order: int, dimension: int, name: str = None) -> tn.Node:
  """Create a delta node with given order and dimension.

  Args:
      order (int): the number of legs of the tensor
      dimension (int): the size of each leg

  Returns:
      tn.Node: delta node

  """
  if name is None:
    name = f"{order} delta"
  delta = jnp.zeros([dimension] * order)

  for i in range(dimension):
    delta[(i,) * order] = 1
  return tn.Node(delta, name)


def random(
  key: jax.Array,
  axes_size: tuple[int],
  rng_function: Callable[[jax.Array], tuple[int]] = jax.random.uniform,
) -> tn.Node:
  """Create a random node.

  Args:
      key (jax.Array): Jax PRNG Key
      axes_size (tuple[int]): Sizes of the axes of the node.
      rng_function (Callable[[jax.Array], tuple[int]]): Function that generates random numbers. Defaults to jax.random.uniform.

  Returns:
      tn.Node: random node

  """
  tensor = rng_function(key, axes_size)
  return tn.Node(tensor)
