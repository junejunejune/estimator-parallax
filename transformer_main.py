# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer import compute_bleu
from official.transformer import translate
from official.transformer.model import model_params
from official.transformer.model import transformer
from official.transformer.utils import dataset
from official.transformer.utils import metrics
from official.transformer.utils import schedule
from official.transformer.utils import tokenizer
from official.utils.accelerator import tpu as tpu_util
from official.utils.export import export
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

from tensorflow.contrib.learn.python.learn import run_config
import collections
import six
import itertools

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}


DEFAULT_TRAIN_EPOCHS = 10
INF = int(1e9)
BLEU_DIR = "bleu"

# Dictionary containing tensors that are logged by the logging hooks. Each item
# maps a string to the tensor name.
TENSORS_TO_LOG = {
    "learning_rate": "model/get_train_op/learning_rate/learning_rate",
    "cross_entropy_loss": "model/cross_entropy"}

from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.training import device_setter
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

from tensorflow_estimator.contrib.estimator.python.estimator import replicate_model_fn
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import training
from tensorflow.python.training import device_setter as device_setter_lib

def local_device_setter(num_devices=1, ps_device_type='cpu', worker_device='/cpu:0', ps_ops=None, ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):

    current_device = pydev.DeviceSpec.from_string(op.device or "") 

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def

    #place sparse variable on parameter server

    if node_def.op in ps_ops and (isinstance(node_def.op, ops.IndexedSlices) or isinstance(node_def.op, sparse_tensor.SparseTensor)):
#    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    #place dense variable on each worker
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "") 
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser

def create_tower_network(model, params, features, labels):
  print("features print: ", features)
  print("labels print: ", labels)
  with tf.variable_scope('forward_and_backward', reuse=False):
    logits = model(features, labels)
    logits.set_shape(labels.shape.as_list() + logits.shape.as_list()[2:])
    xentropy, weights = metrics.padded_cross_entropy_loss(logits, labels, params["label_smoothing"], params["vocab_size"])
    loss = tf.reduce_sum(xentropy)/tf.reduce_sum(weights)
    return logits, loss

from tensorflow.python.ops import array_ops
def split_batch(features, labels, number_of_shards):
  def ensure_divisible_by_shards(sequence):
    batch_size = ops.convert_to_tensor(sequence).get_shape()[0]
    if batch_size % number_of_shards != 0:
      raise ValueError(
          'Batch size {} needs to be divisible by the number of GPUs, which '
          'is {}.'.format(batch_size, number_of_shards))

  def split_dictionary(dictionary):
    """Split a dictionary into shards."""
    shards = [{} for _ in range(number_of_shards)]
    for name, tensor in six.iteritems(dictionary):
      if isinstance(tensor, sparse_tensor.SparseTensor):
        for i, shard in enumerate(
            sparse_ops.sparse_split(
                sp_input=tensor, num_split=number_of_shards, axis=0)):
          shards[i][name] = shard
      else:
        ensure_divisible_by_shards(tensor)
        for i, shard in enumerate(array_ops.split(tensor, number_of_shards)):
          shards[i][name] = shard
    return shards

  if isinstance(features, dict):
    feature_shards = split_dictionary(features)
  else:
    ensure_divisible_by_shards(features)
    feature_shards = array_ops.split(features, number_of_shards)

  if labels is None:
    label_shards = None
  elif isinstance(labels, dict):
    label_shards = split_dictionary(labels)
  else:
    ensure_divisible_by_shards(labels)
    label_shards = array_ops.split(labels, number_of_shards)
  return feature_shards, label_shards


def get_model_fn(train_input_fn, is_chief, batch_size, flags_obj):
 
  def model_fn(features, labels, mode, params):
    """Defines how to train, evaluate and predict from the transformer model."""  

    num_gpus=flags_core.get_num_gpus(flags_obj)
    
    learning_rate = get_learning_rate(learning_rate=params["learning_rate"], hidden_size=params["hidden_size"], learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
    optimizers = [tf.contrib.opt.LazyAdamOptimizer(learning_rate, beta1=params["optimizer_adam_beta1"], beta2=params["optimizer_adam_beta2"], epsilon=params["optimizer_adam_epsilon"]) for _ in range(num_gpus)]

    if params["dtype"] == "fp16":
      optimizers = [tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer) for optimizer in optimizers]
   
#    feature_shards, label_shards = replicate_model_fn._split_batch(features, labels, num_gpus, device=consolidation_device)
#    feature_shards, label_shards = split_batch(features, labels, num_gpus)

    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)
    grad_list= []
    losses = []
    logits = []
    for gpu_idx in range(num_gpus):
      device_setter = local_device_setter(ps_device_type='cpu', worker_device='/gpu:{}'.format(gpu_idx))
      with tf.device(device_setter):
#      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope('tower%d'%gpu_idx):
#with tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster_spec)):
        logit, loss = create_tower_network(model, params, features, labels)
#        feature_shard, label_shard = next(iterator)
#        logit, loss = create_tower_network(model, params, features, labels)
        logits.append(logit)
        losses.append(loss)
        grad_list.append([x for x in optimizers[gpu_idx].compute_gradients(loss) if x[0] is not None])

#    output_train = tf.concat(logits, axis=0)
    output_train = tf.reduce_mean(logits, axis=0)
    loss_train = tf.reduce_mean(losses, name='loss')
    
    grads = []
    all_vars= []
    for tower in grad_list:
      grads.append([x[0] for x in tower])
      all_vars.append([x[1] for x in tower])
    
    reduced_grad = []
#    from tensorflow.contrib import nccl
    from tensorflow.python.ops import nccl_ops
    if num_gpus==1:
      reduced_grad = grads
    else:
      new_all_grads = []
      for grad in zip(*grads):
#        summed = nccl.all_sum(grad)
        summed = nccl_ops.all_sum(grad)
        grads_for_devices = []
        for g in summed:
          with tf.device(g.device):
            g = tf.multiply(g, 1.0 / num_gpus, name='allreduce_avg')
          grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)
      reduced_grad = list(zip(*new_all_grads))
    
#    grads = merge_grad_list(reduced_grad, all_vars)
    grads = [list(zip(gs, vs)) for gs, vs in zip(reduced_grad, all_vars)]

    #apply gradients to each GPU by broadcasting summed gradient
    train_ops = []
    for idx, grad_and_vars in enumerate(grads):
      with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='tower%d'%idx)
        with tf.control_dependencies(update_ops):
          train_ops.append(optimizers[idx].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx)))
    optimize_op = tf.group(*train_ops, name='train_op')
    train_metrics = {"learning_rate": learning_rate}

    tf.identity(loss_train, "cross_entropy")

    if mode == tf.estimator.ModeKeys.TRAIN:
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss_train, train_op=optimize_op)
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss_train, predictions={"predictions": output_train}, eval_metric_ops=metrics.get_eval_metrics(output_train, labels, params))
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=output_train, export_outputs={"translate": tf.estimator.export.PredictOutput(output_train)})

  return model_fn

def _tower_fn(i, model, features, labels, params):

  logits = model(features, labels)

  xentropy, weights = metrics.padded_cross_entropy_loss(logits, labels, params["label_smoothing"], params["vocab_size"])
  tower_loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
  with tf.variable_scope("get_train_op"):
    learning_rate = get_learning_rate(learning_rate=params["learning_rate"], hidden_size=params["hidden_size"], learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate, beta1=params["optimizer_adam_beta1"], beta2=params["optimizer_adam_beta2"], epsilon=params["optimizer_adam_epsilon"])
    model_params = tf.trainable_variables()
#    tower_grad = optimizer.compute_gradients(tower_loss, model_params, colocate_gradients_with_ops=True)
    with tf.variable_scope("is_gradient"): 
      tower_grad = tf.gradients(tower_loss, model_params)
      return tower_loss, zip(tower_grad, model_params), logits

def record_scalars(metric_dict):
  for key, value in metric_dict.items():
    tf.contrib.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  print("def get_learning_rate")
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())
    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
    # Create a named tensor that will be logged using the logging hook.
    # The full name includes variable and names scope. In this case, the name
    # is model/get_train_op/learning_rate/learning_rate
    tf.identity(learning_rate, "learning_rate")

    return learning_rate


def translate_and_compute_bleu(estimator, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      estimator, subtokenizer, bleu_source, output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def get_global_step(estimator):
  """Return estimator's last checkpoint."""
  return int(estimator.latest_checkpoint().split("-")[-1])


def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref, vocab_file):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      estimator, subtokenizer, bleu_source, bleu_ref)

  tf.logging.info("Bleu score (uncased): %d", uncased_score)
  tf.logging.info("Bleu score (cased): %d", cased_score)
  return uncased_score, cased_score


def _validate_file(filepath):
  """Make sure that file exists."""
  if not tf.gfile.Exists(filepath):
    raise tf.errors.NotFoundError(None, None, "File %s not found." % filepath)


def run_loop(
    estimator, schedule_manager, train_hooks=None, benchmark_logger=None,
    bleu_source=None, bleu_ref=None, bleu_threshold=None, vocab_file=None):
  print("def run_loop")
  """Train and evaluate model, and optionally compute model's BLEU score.

  **Step vs. Epoch vs. Iteration**

  Steps and epochs are canonical terms used in TensorFlow and general machine
  learning. They are used to describe running a single process (train/eval):
    - Step refers to running the process through a single or batch of examples.
    - Epoch refers to running the process through an entire dataset.

  E.g. training a dataset with 100 examples. The dataset is
  divided into 20 batches with 5 examples per batch. A single training step
  trains the model on one batch. After 20 training steps, the model will have
  trained on every batch in the dataset, or, in other words, one epoch.

  Meanwhile, iteration is used in this implementation to describe running
  multiple processes (training and eval).
    - A single iteration:
      1. trains the model for a specific number of steps or epochs.
      2. evaluates the model.
      3. (if source and ref files are provided) compute BLEU score.

  This function runs through multiple train+eval+bleu iterations.

  Args:
    estimator: tf.Estimator containing model to train.
    schedule_manager: A schedule.Manager object to guide the run loop.
    train_hooks: List of hooks to pass to the estimator during training.
    benchmark_logger: a BenchmarkLogger object that logs evaluation data
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translations for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.
    vocab_file: Path to vocab file that will be used to subtokenize bleu_source.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
    NotFoundError: if the vocab file or bleu files don't exist.
  """
  if bleu_source:
    _validate_file(bleu_source)
  if bleu_ref:
    _validate_file(bleu_ref)
  if vocab_file:
    _validate_file(vocab_file)

  evaluate_bleu = bleu_source is not None and bleu_ref is not None

  # Print details of training schedule.
  tf.logging.info("Training schedule:")
  tf.logging.info(
      "\t1. Train for {}".format(schedule_manager.train_increment_str))
  tf.logging.info("\t2. Evaluate model.")
  if evaluate_bleu:
    tf.logging.info("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      tf.logging.info("Repeat above steps until the BLEU score reaches %f" %
                      bleu_threshold)
  if not evaluate_bleu or bleu_threshold is None:
    tf.logging.info("Repeat above steps %d times." %
                    schedule_manager.train_eval_iterations)

  if evaluate_bleu:
    # Create summary writer to log bleu score (values can be displayed in
    # Tensorboard).
    bleu_writer = tf.summary.FileWriter(
        os.path.join(estimator.model_dir, BLEU_DIR))
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      schedule_manager.train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  for i in xrange(schedule_manager.train_eval_iterations):
    tf.logging.info("Starting iteration %d" % (i + 1))

    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(
        dataset.train_input_fn,
        steps=schedule_manager.single_iteration_train_steps,
        hooks=train_hooks)
    print("Done estimator training")
    '''
    eval_results = estimator.evaluate(
        input_fn=dataset.eval_input_fn,
        steps=schedule_manager.single_iteration_eval_steps)

    tf.logging.info("Evaluation results (iter %d/%d):" %
                    (i + 1, schedule_manager.train_eval_iterations))
    tf.logging.info(eval_results)
    benchmark_logger.log_evaluation_result(eval_results)

    # The results from estimator.evaluate() are measured on an approximate
    # translation, which utilize the target golden values provided. The actual
    # bleu score must be computed using the estimator.predict() path, which
    # outputs translations that are not based on golden values. The translations
    # are compared to reference file to get the actual bleu score.
    if evaluate_bleu:
      uncased_score, cased_score = evaluate_and_log_bleu(
          estimator, bleu_source, bleu_ref, vocab_file)

      # Write actual bleu scores using summary writer and benchmark logger
      global_step = get_global_step(estimator)
      summary = tf.Summary(value=[
          tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
          tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
      ])
      bleu_writer.add_summary(summary, global_step)
      bleu_writer.flush()
      benchmark_logger.log_metric(
          "bleu_uncased", uncased_score, global_step=global_step)
      benchmark_logger.log_metric(
          "bleu_cased", cased_score, global_step=global_step)

      # Stop training if bleu stopping threshold is met.
      if model_helpers.past_stop_threshold(bleu_threshold, uncased_score):
        bleu_writer.close()
        break
    '''

def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, train_epochs, etc.).
  flags_core.define_base()
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=False,
      all_reduce_alg=True
  )
  flags_core.define_benchmark()
  flags_core.define_device(tpu=True)

  # Set flags from the flags_core module as "key flags" so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name="param_set", short_name="mp", default="big",
      enum_values=PARAMS_MAP.keys(),
      help=flags_core.help_wrap(
          "Parameter set to use when creating and training the model. The "
          "parameters define the input shape (batch size and max length), "
          "model configuration (size of embedding, # of hidden layers, etc.), "
          "and various other settings. The big parameter set increases the "
          "default batch size, embedding/hidden size, and filter size. For a "
          "complete list of parameters, please see model/model_params.py."))

  flags.DEFINE_bool(
      name="static_batch", default=False,
      help=flags_core.help_wrap(
          "Whether the batches in the dataset should have static shapes. In "
          "general, this setting should be False. Dynamic shapes allow the "
          "inputs to be grouped so that the number of padding tokens is "
          "minimized, and helps model training. In cases where the input shape "
          "must be static (e.g. running on TPU), this setting will be ignored "
          "and static batching will always be used."))

  # Flags for training with steps (may be used for debugging)
  flags.DEFINE_integer(
      name="train_steps", short_name="ts", default=None,
      help=flags_core.help_wrap("The number of steps used to train."))
  flags.DEFINE_integer(
      name="steps_between_evals", short_name="sbe", default=1000,
      help=flags_core.help_wrap(
          "The Number of training steps to run between evaluations. This is "
          "used if --train_steps is defined."))

  # BLEU score computation
  flags.DEFINE_string(
      name="bleu_source", short_name="bls", default=None,
      help=flags_core.help_wrap(
          "Path to source file containing text translate when calculating the "
          "official BLEU score. Both --bleu_source and --bleu_ref must be set. "
          "Use the flag --stop_threshold to stop the script based on the "
          "uncased BLEU score."))
  flags.DEFINE_string(
      name="bleu_ref", short_name="blr", default=None,
      help=flags_core.help_wrap(
          "Path to source file containing text translate when calculating the "
          "official BLEU score. Both --bleu_source and --bleu_ref must be set. "
          "Use the flag --stop_threshold to stop the script based on the "
          "uncased BLEU score."))
  flags.DEFINE_string(
      name="vocab_file", short_name="vf", default=None,
      help=flags_core.help_wrap(
          "Path to subtoken vocabulary file. If data_download.py was used to "
          "download and encode the training data, look in the data_dir to find "
          "the vocab file."))

  flags_core.set_defaults(data_dir="/tmp/translate_ende",
                          model_dir="/tmp/transformer_model",
                          batch_size=None,
                          train_epochs=None)

  @flags.multi_flags_validator(
      ["train_epochs", "train_steps"],
      message="Both --train_steps and --train_epochs were set. Only one may be "
              "defined.")
  def _check_train_limits(flag_dict):
    return flag_dict["train_epochs"] is None or flag_dict["train_steps"] is None

  @flags.multi_flags_validator(
      ["bleu_source", "bleu_ref"],
      message="Both or neither --bleu_source and --bleu_ref must be defined.")
  def _check_bleu_files(flags_dict):
    return (flags_dict["bleu_source"] is None) == (
        flags_dict["bleu_ref"] is None)

  @flags.multi_flags_validator(
      ["bleu_source", "bleu_ref", "vocab_file"],
      message="--vocab_file must be defined if --bleu_source and --bleu_ref "
              "are defined.")
  def _check_bleu_vocab_file(flags_dict):
    if flags_dict["bleu_source"] and flags_dict["bleu_ref"]:
      return flags_dict["vocab_file"] is not None
    return True

  @flags.multi_flags_validator(
      ["export_dir", "vocab_file"],
      message="--vocab_file must be defined if --export_dir is set.")
  def _check_export_vocab_file(flags_dict):
    if flags_dict["export_dir"]:
      return flags_dict["vocab_file"] is not None
    return True

  flags_core.require_cloud_storage(["data_dir", "model_dir", "export_dir"])


import functools
def get_experiment_fn(is_chief, flags_obj, params, schedule_manager, num_gpus, variable_strategy, use_distortion_for_training=True):
  def _experiment_fn(run_config, hparams):
#    train_input_fn = functools.partial(
#        dataset.train_input_fn, params["data_dir"], batch_size=schedule_manager.batch_size, use_distortion_for_training=use_distortion_for_training)
#    eval_input_fn = functools.partial(
#        dataset.eval_input_fn, params["data_dir"], batch_size=schedule_manager.batch_size, num_shards=num_gpus)
    train_input_fn = dataset.train_input_fn
    eval_input_fn = dataset.eval_input_fn
    train_steps = flags_obj.train_steps
    eval_steps = flags_obj.steps_between_evals

#    estimator = construct_estimator(flags_obj, params, schedule_manager)
    estimator = tf.estimator.Estimator(model_fn=get_model_fn(train_input_fn, is_chief,params["batch_size"], flags_obj), model_dir=flags_obj.model_dir, params=params, config=run_config)
    return tf.contrib.learn.Experiment(
        estimator, train_input_fn=dataset.train_input_fn, eval_input_fn=dataset.eval_input_fn, train_steps=train_steps, eval_steps=eval_steps)
  return _experiment_fn 

class RunConfig(tf.contrib.learn.RunConfig):
  def uid(self, whitelist=None):
    if whitelist is None:
      whitelist = run_config._DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    for k in whitelist:
      state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
         sorted(ordered_state['_cluster_spec'].as_dict().items(),
                key=lambda t: t[0])
      )   

    print("IN RUNCONFIG")
    print(', '.join('%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))) 
    return ', '.join('%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state)) 

def run_transformer(flags_obj):
  print("run_transformer")
  """Create tf.Estimator to train and evaluate transformer model.

  Args:
    flags_obj: Object containing parsed flag values.
  """
  num_gpus = flags_core.get_num_gpus(flags_obj)
  print("NUM_GPUS: ", num_gpus)

  # Add flag-defined parameters to params object
  params = PARAMS_MAP[flags_obj.param_set]
  if num_gpus > 1:
    if flags_obj.param_set == "big":
      params = model_params.BIG_MULTI_GPU_PARAMS
    elif flags_obj.param_set == "base":
      params = model_params.BASE_MULTI_GPU_PARAMS
  
  params["num_gpus"] = num_gpus
  params["data_dir"] = flags_obj.data_dir
  params["model_dir"] = flags_obj.model_dir
  params["num_parallel_calls"] = flags_obj.num_parallel_calls

  params["tpu"] = flags_obj.tpu
  params["static_batch"] = flags_obj.static_batch
  params["allow_ffn_pad"] = True

  params["use_synthetic_data"] = flags_obj.use_synthetic_data

  # Set batch size parameter, which depends on the availability of
  # TPU and GPU, and distribution settings.
  params["batch_size"] = (flags_obj.batch_size or params["default_batch_size"])
  print("BATCH_SIZE1: ",params["batch_size"])
#  params["batch_size"] = distribution_utils.per_device_batch_size(params["batch_size"], num_gpus)
#  print("BATCH_SIZE2: ", params["batch_size"])

  schedule_manager = schedule.Manager(
      train_steps=flags_obj.train_steps,
      steps_between_evals=flags_obj.steps_between_evals,
      train_epochs=flags_obj.train_epochs,
      epochs_between_evals=flags_obj.epochs_between_evals,
      default_train_epochs=DEFAULT_TRAIN_EPOCHS,
      batch_size=params["batch_size"],
      max_length=params["max_length"],
      use_tpu=False,
      num_tpu_shards=flags_obj.num_tpu_shards
  )

  params["repeat_dataset"] = schedule_manager.repeat_dataset

  model_helpers.apply_clean(flags.FLAGS)

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      tensors_to_log=TENSORS_TO_LOG,  # used for logging hooks
      batch_size=schedule_manager.batch_size,  # for ExamplesPerSecondHook
      use_tpu=False  # Not all hooks can run with TPUs
  )
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="transformer",
      dataset_name="wmt_translate_ende",
      run_params=params,
      test_id=flags_obj.benchmark_test_id)

  # Train and evaluate transformer model
#  estimator = construct_estimator(flags_obj, params, schedule_manager)

  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      intra_op_parallelism_threads=0,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))
  sess_config.gpu_options.allocator_type = 'BFC'
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
  print("SESS_CONFIG: ", sess_config)
  config = RunConfig(session_config=sess_config, model_dir=params["model_dir"])
  variable_strategy = 'GPU'
  use_distortion_for_training=True
  experiment_fn = get_experiment_fn(config.is_chief, flags_obj, params, schedule_manager, num_gpus, variable_strategy, use_distortion_for_training)
	
#tf.contrib.learn.learn_runner.run(experiment_fn, run_config=config, hparams=tf.contrib.training.HParams(is_chief=config.is_chief, **hparams))
 
  tf.contrib.learn.learn_runner.run(experiment_fn, run_config=config, hparams=tf.contrib.training.HParams(is_chief=config.is_chief, **params))
  
  '''
  run_loop(
      estimator=estimator,
      # Training arguments
      schedule_manager=schedule_manager,
      train_hooks=train_hooks,
      benchmark_logger=benchmark_logger,
      # BLEU calculation arguments
      bleu_source=flags_obj.bleu_source,
      bleu_ref=flags_obj.bleu_ref,
      bleu_threshold=flags_obj.stop_threshold,
      vocab_file=flags_obj.vocab_file)
  '''

  if flags_obj.export_dir:
    serving_input_fn = export.build_tensor_serving_input_receiver_fn(
        shape=[None], dtype=tf.int64, batch_size=None)
    # Export saved model, and save the vocab file as an extra asset. The vocab
    # file is saved to allow consistent input encoding and output decoding.
    # (See the "Export trained model" section in the README for an example of
    # how to use the vocab file.)
    # Since the model itself does not use the vocab file, this file is saved as
    # an extra asset rather than a core asset.
    estimator.export_savedmodel(
        flags_obj.export_dir, serving_input_fn,
        assets_extra={"vocab.txt": flags_obj.vocab_file},
        strip_default_attrs=True)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_transformer(flags.FLAGS)


if __name__ == "__main__":
  
  tf.enable_eager_execution()
  tf.logging.set_verbosity(tf.logging.INFO)
  define_transformer_flags()
  absl_app.run(main)
