from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.models import Model,load_model
import keras.layers as KL
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import h5py
import keras.backend as K

try:
    from keras.engine import saving
except ImportError:
    # Keras before 2.2 used the 'topology' namespace.
    from keras.engine import topology as saving
# f=h5py.File('/home/li/Mask_RCNN_point/mobilenet_v2_weights.h5','r')
# if 'layer_names' not in f.attrs and 'model_weights' in f:
#     f = f['model_weights']
# layers=[n.decode('utf8') for n in f.attrs['layer_names']]
# print(layers)
# # for key in f.keys():
# #     print(f[key].name)
# #     # print(f[key].shape)

def load_weights_from_hdf5_group_by_name(f, layers):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = saving.preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '") expects ' +
                                 str(len(symbolic_weights)) +
                                 ' weight(s), but the saved weights' +
                                 ' have ' + str(len(weight_values)) +
                                 ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))
    K.batch_set_value(weight_value_tuples)
    return weight_value_tuples

def load_weights(model, filepath, by_name=False, exclude=None):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    if exclude:
        by_name = True

    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    keras_model = model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        else keras_model.layers

    # Exclude some layers
    if exclude:
        layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
        summary = load_weights_from_hdf5_group_by_name(f, layers)
        print(len(summary))
    else:
        saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
        f.close()


input_image = KL.Input(
            shape=[448, 448, 3], name="input_image")
# print(input_image.shape)
model = MobileNetV2(input_tensor=input_image, alpha=1.0, weights= 'imagenet',include_top=False)
# model = MobileNetV2((96,96,3),include_top=False)
C1 = model.get_layer('expanded_conv_project_BN').output
C2 = model.get_layer('block_2_add').output
C3 = model.get_layer('block_5_add').output
C4 = model.get_layer('block_12_add').output
C5 = model.get_layer('block_16_project_BN').output
model1 = Model(model.input,[C1,C2,C3,C4,C5])
train_bn = False
if not train_bn:
    for layers in model1.layers:
        if 'BN' in layers.name:
            layers.trainable = False
# filepath='/home/li/Mask_RCNN_point/mobilenet_v2_weights.h5'
# load_weights(model1, filepath, by_name=True)
# model = MobileNetV2(input_shape=(512,512,3),weights= 'imagenet',include_top=False)
for layers in model1.layers:
    print(layers.name)
    print(layers.output)