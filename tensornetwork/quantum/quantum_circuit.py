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
File: tensornetwork/quantum/quantum_circuit.py
Contact: oskar.pfeffer@ptb.de
Github: https://github.com/oskarpfeffer/TensorNetwork
Description: Methods to transform TN into quantum circuits and to handle quantum circuits.
"""

import tensornetwork as tn
import jax
import jax.numpy as jnp
import tensornetwork.quantum.utils as utils
import tensornetwork.base_elements.nodes as nodes


def block_encode_matrix(
  tensor: jax.Array, normalize: bool = False, to_last_axes: bool = False
):
  """Block-encodes n*n tensor in a 2n*2n unitary.

  Matrix will be encoded as c * [[M, sqrt(1 - M M.dag)], [sqrt(1 - M.dag M)], -M.dag], where c == (1 or 1/||tensor||_2) is a normalization constant.
  Can be broadcasted via standard broadcasting rules, then the normalization used is where c == max(1/||matrix||_2) over all matrices.

  Args:
      tensor (jax.Array): Matrix to be encoded
      normalize (bool): If True the block encoded matrix will be M/||M||_2. Defaults to False.
      to_last_axes (bool): If True returns the matrix as a (n,2,n,2).reshape(2*n, 2*n) tensor with block-encoding on the last in/out/axis. Defaults to False.
  Returns:
      jax.Array: Unitary that block-encodes M

  """
  n = tensor.shape[-1]

  if normalize:
    c = jnp.linalg.matrix_norm(tensor, ord=2).max()
    tensor = tensor / jnp.nextafter(c, jnp.inf)
    # jnp.nextafter to ensure that ||tensor|| >= 1

  # U = [[M, u sqrt(1 - s^2) u*], [v* sqrt(1 - s^2) v*, -M*]
  u, s, v = jnp.linalg.svd(tensor)
  Block01 = (
    u
    @ jnp.apply_along_axis(jnp.diag, -1, jnp.sqrt(1 - s**2))
    @ jnp.matrix_transpose(u.conj())
  )
  Block10 = (
    jnp.matrix_transpose(v.conj())
    @ jnp.apply_along_axis(jnp.diag, -1, jnp.sqrt(1 - s**2))
    @ v
  )
  Block11 = jnp.matrix_transpose(-tensor.conj())

  U = jnp.empty(tensor.shape[:-2] + (2 * n, 2 * n))
  U = U.at[..., :n, :n].set(tensor)
  U = U.at[..., :n, n:].set(Block01)
  U = U.at[..., n:, :n].set(Block10)
  U = U.at[..., n:, n:].set(Block11)
  if to_last_axes:
    U = U.reshape(U.shape[:-2] + (2, n, 2, n))
    U = U.transpose(tuple(range(U.ndim - 4)) + (-3, -4, -1, -2))
    U = U.reshape(U.shape[:-4] + (2 * n, 2 * n))

  return U


def block_encode_tensor(
  tensor: jax.Array, in_axes_index: tuple = (-2,), out_axes_index: tuple = (-1,)
):
  """Makes operator unitary along given axes via block-encoding.

  If the tensor is a matrix M, then it will be encoded as c * [[M, sqrt(1 - M M.dag)], [sqrt(1 - M.dag M)], -M.dag], where c = 1/||M||_2 is a normalization constant. If the node is a tensor of order > 2, each individual matrix along the given dimension will be block encoded, however the normalization constant will be equal to the largest singular value of all matrices.

  Args:
      tensor: tensor to be block encoded
      in_axes_index (tuple): input_axes of the operator
      out_axes_index (tuple): output axes of the operator

  Returns:
      jax.Array: Block encoded tensor.
  """
  in_axes_index = utils.get_positive_index(tensor, in_axes_index)
  out_axes_index = utils.get_positive_index(tensor, out_axes_index)
  in_axes_size = utils.get_axes_size(tensor, in_axes_index)
  out_axes_size = utils.get_axes_size(tensor, out_axes_index)
  tensor = utils.transpose_join_computational_axes(
    tensor, in_axes_index, out_axes_index
  )
  tensor = block_encode_matrix(tensor, normalize=True, to_last_axes=True)
  in_axes_size += (2,)
  out_axes_size += (2,)
  in_axes_index += (tensor.ndim,)
  out_axes_index += (tensor.ndim + 1,)
  tensor = utils.transpose_join_computational_axes(
    tensor, in_axes_index, out_axes_index, in_axes_size, out_axes_size, invert=True
  )
  assert utils.is_unitary(tensor, in_axes_index, out_axes_index)
  return tensor


def block_encode_node(
  node: tn.Node, in_axes_index: tuple[int], out_axes_index: tuple[int]
) -> list[tn.Node]:
  """_summary_

  Args:
      node (tn.Node): _description_
      in_axes_index (tuple[int]): _description_
      out_axes_index (tuple[int]): _description_

  Returns:
      list[tn.Node]: _description_
  """
  edges = node.edges
  rank = node.get_rank()
  node.tensor = block_encode_tensor(node.tensor, in_axes_index, out_axes_index)
  edges += [tn.Edge(node, rank), tn.Edge(node, rank + 1)]
  node.edges = edges
  aux1 = nodes.zero_state(node.get_dimension(-2))
  aux2 = nodes.zero_state(node.get_dimension(-2))
  node[-2] ^ aux1[0]
  node[-1] ^ aux2[0]
  return [node, aux1, aux2]


def split_edge_control(edge: tn.Edge) -> list[tn.Node]:
  """Replaces an edge with two controls and two all-1 vectors to implement the
  contraction on a quantum computer.

  Args:
      edge (tn.Edge): The edge that needs to be replaces

  Returns:
      set(tn.Node): Set with the two initial and all the new created nodes.

  """
  node1, node2 = edge.get_nodes()
  axis1 = edge.axis1
  axis2 = edge.axis2
  dimension = edge.dimension
  edge.disconnect()
  edge.disable()
  control1 = nodes.delta_node(3, dimension)
  control2 = nodes.delta_node(3, dimension)
  top_vector = nodes.delta_node(1, dimension)
  bottom_vector = nodes.delta_node(1, dimension)

  node1[axis1] ^ control1[0]
  control1[2] ^ control2[0]
  control1[1] ^ top_vector[0]
  control2[1] ^ node2[axis2]
  control2[2] ^ bottom_vector[0]

  return [node1, control1, control2, node2, top_vector, bottom_vector]
