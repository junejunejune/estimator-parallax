

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

import functools


def get_model_fn(is_chief, batch_size, flags_obj):
  def model_fn(features, labels, mode, params):
    """Defines how to train, evaluate and predict from the transformer model."""  
    num_devices=flags_core.get_num_gpus(flags_obj)
    consolidation_device = 'gpu:0'
    feature_shards, label_shards = replicate_model_fn._split_batch(features, labels, num_devices, device=consolidation_device)

    tower_losses = []
    tower_gradvars = []
    tower_preds = []
    for i in range(num_devices):
      worker_device = '/{}:{}'.format('gpu', i)
      device_setter = local_device_setter(ps_device_type='cpu', worker_device=worker_device, ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(num_devices, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('model', reuse=bool(i != 0)):
        model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)
        
        #Per-model separate variables
        with tf.name_scope('tower_%d' % i) as name_scope:  
          with tf.device(device_setter):
            print("CURRENT_SCOPE", tf.get_variable_scope().name)
            import sys
            sys.exit(1)
              # Create model and get output logits.
            loss, gradvars, preds = create_network_for_tower_fn(model, feature_shards[i], label_shards[i], params=params)
        tower_losses.append(loss)
        tower_gradvars.append(gradvars)
        tower_preds.append(preds)
    
    # Compute global loss and gradients
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
#          print("AVG_GRAD: ", avg_grad, "VAR: ", var) 
          gradvars.append((avg_grad, var))

    with tf.device(consolidation_device):
      loss = tf.reduce_mean(tower_losses, name='loss')
      tf.identity(loss, "cross_entropy")
      logits = tf.concat([l for l in tower_preds], axis=0)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=logits, export_outputs={"translate": tf.estimator.export.PredictOutput(logits)})

      if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope("get_train_op"):
          print("in get_train_op")
          learning_rate = get_learning_rate(learning_rate=params["learning_rate"], hidden_size=params["hidden_size"], learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
          optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate, beta1=params["optimizer_adam_beta1"], beta2=params["optimizer_adam_beta2"], epsilon=params["optimizer_adam_epsilon"])
          optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=num_devices)
          sync_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)
#          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

          global_step = tf.train.get_global_step()
          update_ops = tf.assign(global_step, global_step + 1, name='update_global_step')
          minimize_op = optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())
          train_op = tf.group(minimize_op, update_ops)
          #train_op = [optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())]
          metric_dict = {"learning_rate": learning_rate}
          metric_dict["minibatch_loss"] = loss
          record_scalars(metric_dict)
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss, training_hooks=[sync_hook], train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions={"predictions": logits}, eval_metric_ops=metrics.get_eval_metrics(logits, labels,params))      
  return model_fn



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
    estimator = tf.estimator.Estimator(model_fn=get_model_fn(is_chief,schedule_manager.batch_size, flags_obj), model_dir=flags_obj.model_dir, params=params, config=run_config)
    return tf.contrib.learn.Experiment(
        estimator, train_input_fn=dataset.train_input_fn, eval_input_fn=dataset.eval_input_fn, train_steps=train_steps, eval_steps=eval_steps)
  return _experiment_fn 



def run_transformer(flags_obj):
  print("def run_transformer")
  """Create tf.Estimator to train and evaluate transformer model.

  Args:
    flags_obj: Object containing parsed flag values.
  """
  num_gpus = flags_core.get_num_gpus(flags_obj)

  # Add flag-defined parameters to params object
  params = PARAMS_MAP[flags_obj.param_set]
  if num_gpus > 1:
    if flags_obj.param_set == "big":
      params = model_params.BIG_MULTI_GPU_PARAMS
    elif flags_obj.param_set == "base":
      params = model_params.BASE_MULTI_GPU_PARAMS

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

  params["batch_size"] = distribution_utils.per_device_batch_size(
      params["batch_size"], num_gpus)

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

  # Train and evaluate transformer model
#  estimator = construct_estimator(flags_obj, params, schedule_manager)

  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True,
      intra_op_parallelism_threads=0,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  print("SESS_CONFIG: ", sess_config)
  config = RunConfig(session_config=sess_config, model_dir=params["model_dir"])
  variable_strategy = 'CPU'
  use_distortion_for_training=True
  experiment_fn = get_experiment_fn(config.is_chief, flags_obj, params, schedule_manager, num_gpus, variable_strategy, use_distortion_for_training)
	
#tf.contrib.learn.learn_runner.run(experiment_fn, run_config=config, hparams=tf.contrib.training.HParams(is_chief=config.is_chief, **hparams))
 
  tf.contrib.learn.learn_runner.run(experiment_fn, run_config=config, hparams=tf.contrib.training.HParams(is_chief=config.is_chief, **params))

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
  print("main")
  run_transformer(flags.FLAGS)

if __name__ == "__main__":
  absl_app.run(main)
