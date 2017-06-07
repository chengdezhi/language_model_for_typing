import math
import tensorflow as tf


def linear(x, size, name):
    w = tf.get_variable(name + "/W", [x.get_shape()[-1], size])
    b = tf.get_variable(name + "/b", [1, size], initializer=tf.zeros_initializer)
    return tf.matmul(x, w) + b


def sharded_variable(name, shape, num_shards, dtype=tf.float32, transposed=False):
    # The final size of the sharded variable may be larger than requested.
    # This should be fine for embeddings.
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    if transposed:
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    else:
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype)
    return [tf.get_variable(name + "_%d" % i, [shard_size, shape[1]], initializer=initializer, dtype=dtype)
            for i in range(num_shards)]


# XXX(rafal): Code below copied from rnn_cell.py
def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(tf.get_variable(name + "_%d" % i, [current_size] + shape[1:], dtype=dtype))
    return shards


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    _sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(_sharded_variable) == 1:
        return _sharded_variable[0]

    return tf.concat(_sharded_variable, 0)


class LSTMCell(tf.contrib.rnn.BasicRNNCell):

    def __init__(self, num_units, input_size, initializer=None, num_proj=None, num_shards=1, dtype=tf.float32):
        self._num_units = num_units   # 2048  for state_size
        self._initializer = initializer
        self._num_proj = num_proj
        self._num_unit_shards = num_shards
        self._num_proj_shards = num_shards
        self._forget_bias = 1.0
        
        # num_proj or num_units
        if num_proj:
            self._state_size = num_units + num_proj   # 2048 + 512 
            self._output_size = num_proj              # 512 
        else:
            self._state_size = 2 * num_units     
            self._output_size = num_units

        with tf.variable_scope("LSTMCell"):
            # shard for variable   but shard = 1
            # input_size: embedding_size
            # 
            self._concat_w = _get_concat_variable(
                "W", [input_size + num_proj, 4 * self._num_units],
                dtype, self._num_unit_shards)

            # anothet structure for lstm? and what is it:
            self._b = tf.get_variable(
                "B", shape=[4 * self._num_units],dtype=dtype)

            self._concat_w_proj = _get_concat_variable(
                "W_P", [self._num_units, self._num_proj],
                dtype, self._num_proj_shards)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        num_proj = self._num_units if self._num_proj is None else self._num_proj  # 512 for train 
        # state {batch_size, state_size(2048) + proj_size(512)}
        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])          # is c{j-1}
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])   # is h{j-1}

        input_size = inputs.get_shape().with_rank(2)[1]
        print "input_size:",input_size
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with tf.variable_scope(type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = tf.concat([inputs, m_prev],1)    
            lstm_matrix = tf.nn.bias_add(tf.matmul(cell_inputs, self._concat_w), self._b)
            i, j, f, o = tf.split(lstm_matrix,4,1)

            c = tf.sigmoid(f + 1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j)
            m = tf.sigmoid(o) * tf.tanh(c)
            #above  f+1 besides,the standard lstm

            # proj standfor that m is different from c 
            if self._num_proj is not None:
                m = tf.matmul(m, self._concat_w_proj)
        # state size [ba, state_size+proj_size]
        new_state = tf.concat([c, m],1)
        return m, new_state
