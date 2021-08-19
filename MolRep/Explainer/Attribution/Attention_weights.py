
import abc

class AttributionTechnique(abc.ABC):
    """Abstract class for an attribution technique."""

    name: Text
    sample_size: int  # Number of graphs to hold in memory per input.

    @abc.abstractmethod
    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Compute GraphTuple with node and edges importances.
        Assumes that x (GraphTuple) has node and edge information as 2D arrays
        and the returned attribution will be a list of GraphsTuple, for each
        graph inside of x, with the same shape but with 1D node and edge arrays.
        Args:
          x: Input to get attributions for.
          model: model that gives gradients, predictions, activations, etc.
          task_index: index for task to focus attribution.
          batch_index: index for example to focus attribution.
        """


class AttentionWeights(AttributionTechnique):
    """Use attention weights as importance features.
      AttentionWeights uses the attention weights from multi-headead
      self-attention GNN blocks. The weights are on edges and are normalized via
      softmax. We reduce attention on all heads and all blocks to arrive to a
      single value for each edge. These value can be interpreted as importance
      values on the connectivity of a graph.
      Based on "Graph Attention Networks" (https://arxiv.org/abs/1710.10903) and
      "GNNExplainer: Generating Explanations for Graph Neural Networks"
      (https://arxiv.org/pdf/1903.03894.pdf).
    """

    def __init__(self,
                 head_reducer: Callable[...,
                                        tf.Tensor] = tf.math.reduce_mean,
                 block_reducer: Callable[...,
                                         tf.Tensor] = tf.math.reduce_mean,
                 name: Optional[Text] = None):
        """Init.
        Args:
          head_reducer: function used to combine attention weights from each
            attention head in a block.
          block_reducer: function used to combine attention weights across blocks.
          name: name for module.
        """
        self.name = name or self.__class__.__name__
        self.sample_size = 1
        self.head_reducer = head_reducer
        self.block_reducer = block_reducer

    def attribute(self,
                  x: GraphsTuple,
                  model: TransparentModel,
                  task_index: Optional[int] = None,
                  batch_index: Optional[int] = None) -> List[GraphsTuple]:
        """Gets attribtutions."""
        weights = model.get_attention_weights(x)
        weights = tf.stack(weights)  # [n_blocks, n_edges, n_heads]
        weights = self.head_reducer(weights, axis=2)  # [n_blocks, n_edges]
        weights = self.block_reducer(weights, axis=0)  # [n_edges]
        empty_nodes = tf.zeros(len(x.nodes))
        graphs = x.replace(nodes=empty_nodes, edges=weights, globals=None)
        return list(graph_utils.split_graphs_tuple(graphs))