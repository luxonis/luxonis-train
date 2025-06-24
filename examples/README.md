# Extending the Framework

The `luxonis-train` framework is designed to be easily extendable. This document describes how to create custom nodes, losses, metrics, and visualizers.

## Table of Contents

- [Nodes](#nodes)
  - [`BaseHead` Interface](#basehead-interface)
  - [Custom Tasks](#custom-tasks)
    - [Metadata](#metadata)
  - [Node Examples](#node-examples)
    - [ResNet Backbone](#resnet-backbone)
    - [Segmentation Head](#segmentation-head)
- [Custom Attached Modules](#custom-attached-modules)
  - [Automatic Inputs and Labels Extraction](#automatic-inputs-and-labels-extraction)
  - [Attached Modules Examples](#attached-modules-examples)
    - [Simple Loss](#simple-loss)
    - [Complex Loss](#complex-loss)
    - [Metric](#metric)
    - [Visualizer](#visualizer)
- [Testing Custom Components](#testing-custom-components)

## Nodes

Nodes are the main building blocks of the network. Nodes are usually one of the following types:

- Backbone
  - Receives the input image and usually produces a list of feature maps
- Neck
  - Sits on top of the backbone and processes the feature maps
- Head
  - Produces the final output of the network

Backbone and necks should inherit from `BaseNode` class, while heads should inherit from `BaseHead` class.
`BaseHead` offers extended interface on top of `BaseNode` that is used when the model is exported to NN Archive.

To make the most use out of the framework, the nodes should define the following class attributes:

- `attach_index: int | tuple[int, int] | tuple[int, int, int] | Literal["all"]` - specifies which output of the previous node to use
  - Can be either a single integer (negative indexing is supported), a tuple of integers (slice), or the string `"all"`
  - Typically used for heads that are usually connected to backbones producing a list of feature maps
  - If not specified, it is inferred from the type signature of the `forward` method (if possible)
    - Up to debate whether this is a good idea as it's a quite implicit, the reasoning for this is to make implementing custom models as easy as possible with little boilerplate
    - Most of these implicit deductions are logged (and eventually it will be all of them) and I'm gradually improving the error messages so they are as explicit as possible, so it shouldn't be too confusing
- `task: Task` - specifies the task that the node is used for
  - Relevant for heads only
  - Provides better error messages, compatibility checking, more powerful automation, _etc._
  - We offer a set of predefined tasks living in the `Tasks` namespace
    - `CLASSIFICATION` - classification tasks
    - `SEGMENTATION` - segmentation tasks
    - `BOUNDINGBOX` - object detection tasks
    - `INSTANCE_SEGMENTATION` - instance segmentation tasks, requires both `"boundingbox"` and `"segmentation"` labels'
    - `INSTANCE_KEYPOINTS` - instance segmentation tasks, requires both `"boundingbox"` and `"keypoints"` labels
    - `KEYPOINTS` - simple keypoint tasks (2D or 3D pointcloud)
    - `EMBEDDINGS` - used for embedding tasks
    - `ANOMALY_DETECTION` - image anomaly detection tasks
    - `OCR` - optical character recognition
    - `FOMO` - used for the FOMO task. Special task learning on `"boundingbox"` labels, but predicting keypoints
    - This namespace pattern could be a bit confusing if you look into the code. It is supposed to look like an enum because it esentially should be an enum. The only reason it's not is because enum cannot be extended on the user side but we need to support defining custom tasks
  - To define a custom task, see [Custom Tasks](#custom-tasks)

`BaseNode` implements a few convenient properties that can be used to access information about the model:

- `in_channels: int | list[int]` - number of input channels
  - The output is either a single integer or a list of integers depending on the value of `attach_index`
    - That is, if the node is attached to a backbone producing a list of feature maps and the value of `attach_index` is set to `"all"`, `in_channels` will be a list of the channel counts of each feature map
  - Works only if the `attach_index` is defined (or it was possible to infer it)
- `in_width: int | list[int]` - width(s) of the input(s) to the node
- `in_height: int | list[int]` - height(s) of the input(s) to the node
- `n_classes: int` - number of classes
- `n_keypoints: int` - number of keypoints (if the dataset contains keypoint labels)
- `class_names: list[str]` - list of class names
- `original_in_shape: torch.Size` - shape of the original input image
  - Useful for segmentation heads that need to upsample the output to the original image size

> [!TIP]
> You can add a class-level type hint to `in_channels`, `in_width`, and `in_height`. This will cause the values to be checked at initialization time and an exception will be raised if the annotation is incompatible with the outputs of the preceding node. (_e.g._ setting `attach_index` to `"all"` and annotating `in_channels` as `int` will raise an exception)

The main methods of the node are:

- `__init__` - constructor
  - Should always take `**kwargs` as an argument and pass it to the parent constructor
  - All the arguments under `node.params` in the config file are passed here
- `forward(x: T) -> K` - the forward pass of the node
  - In most cases should take either a single tensor or a list of tensors and return again a single tensor or a list of tensors
    - If more control is needed, see the `unwrap` method
- `wrap(outputs: K) -> Packet[Tensor]` - called after `forward`, wraps the output of the node into a dictionary
  - The results of `forward` are not the final outputs of the node, but are wrapped into a dictionary (called a `Packet`)
  - The keys of the dictionary are used to extract the correct values in the attached modules (losses, metrics, visualizers)
  - Usually needs to be overridden for heads only
    - Typically it behaves differently for `train`, `eval`, and `export` calls
    - `train` goes to the loss, `eval` goes to the loss, metrics and visualizers, and `export` is used when the model is exported to ONNX
    - (all of them are also sent to the next node)
  - The default implementation roughly behaves like this:
    - For backbones and necks, the output is wrapped into a dictionary with a single key `"features"`
    - For heads, the output is wrapped into a dictionary with a single key equivalent to the value of `node.task.main_output` property
      - If task is not defined, the node is considered to be a backbone or a neck (_i.e._ using the `"features"` key)
    - Roughly equivalent to:
      ```python
      def wrap(self, output: ForwardOutputType) -> Packet[Tensor]:
          if self.task is not None:
              return {self.task.main_output: output}
          return {"features": output}
      ```
- `unwrap(inputs: list[Packet[Tensor]]) -> T` - called before `forward`, the output of `unwrap` is directly passed to the `forward` method
  - Usually doesn't need to be overridden
  - Receives a list of packets, one for each connected node
    - Usually only one packet is passed
    - Multiple packets are passed if the current node is connected to multiple preceding nodes
      - No such architecture currently implemented in the framework
  - The default implementation looks for a key named `"features"` in the input dictionary and returns its value based on the `attach_index`
    - Roughly equivalent to:
      ```python
      def unwrap(self, inputs: list[Packet[Tensor]]) -> ForwardInputType:
          return inputs[0]["features"][self.attach_index]
      ```
  - Unless the node is connected to a complex backbone producing multiple outputs on top of the feature maps or to another head, this method doesn't need to be overridden

### `BaseHead` Interface

On top of the `BaseNode` interface, `BaseHead` defines the following additional class attributes:

- `parser: str | None` - specifies the parser that should be used with this head
  - _e.g._ `"SegmentationParser"`

The `BaseHead` also defines the following methods that should be overridden:

- `get_custom_head_config() -> dict` - returns a dictionary with custom head configuration
  - Used to populate `head.metadata` field in the NN Archive configuration file

### Custom Tasks

If you need to implement a node that does not fit any of the predefined tasks, you can define a custom task by subclassing the `Task` class. The custom class needs to define the following abstract properties:

- `required_labels: set[str | Metadata]` - set of required labels, can be either a string or a `Metadata` object. For details on `Metadata`, see [Metadata](#metadata)

Additionally, you can override the following properties as well:

- `main_output: str` - specifies the main output of the node
  - Defaults to the name of the task
  - Only relevant for tasks that produce multiple outputs where one can be considered the main output
  - Used to automatically extract the correct values from the node output and dataset (see [Automatic Inputs and Labels Extraction](#automatic-inputs-and-labels-extraction))

#### Metadata

`Metadata` specifies a custom `metadata` label in the dataset. By definition, the metadata labels can have arbitrary names and can be of type `str`, `int`, `float`, or `luxonis_ml.data.Category` (special subclass of `str` for categorical values). The `Metadata` class is used to define the expected structure of the metadata label.

The `Metadata` constructor takes the following arguments:

- `name: str` - expected name of the label
- `typ: type | UnionType` - expected type, supports unions of types

**Example of a Custom Task:**

```python
from luxonis_train.tasks import Task, Metadata


class DistanceEstimation(Task):
    def __init__(self):
        super().__init__("distance")

    @property
    def required_labels(self) -> set[str | Metadata]:
        return {"boundingbox", Metadata("distance", float | int)}

    @property
    def main_output(self) -> str:
        return "boundingbox"

```

The above example could be simplified by inheriting from the `luxonis_train.tasks.BoundingBox` task and overriding the `required_labels` property.

```python
from luxonis_train.tasks import BoundingBox, Metadata

class DistanceEstimation(BoundingBox):
    @property
    def required_labels(self) -> set[str | Metadata]:
        return super().required_labels | {Metadata("distance", float | int)}

```

### Custom Node Examples

#### ResNet Backbone

```python

class ResNet(BaseNode[Tensor, list[Tensor]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        return outputs


```

#### Segmentation Head

```python
from typing import override

from torch import Tensor, nn

from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks

class CustomSegmentationHead(BaseHead):

    # If the head is attached to a node that produces
    # multiple outputs (like a list of feature maps for
    # most bakcbones), this specifies which output to use.
    # Most heads are ususally attached to the last output.
    # If the value is not defined, it is inferred from the
    # type signature of the `forward` method. If that is
    # not possible, some parts of the framework will not
    # be accessible (properties like `in_channels`, `in_width`, etc.)
    # For subclasses of `BaseHead`, the value is automatically set
    # to -1, which means the last output from the previous node
    # (typically the last feature map from the backbone).
    attach_index = -1

    # The `in_channels` property returns either an `int` or
    # a `list[int]` depending on the value of `attach_index`.
    # By specifying its type here, the constructor of `BaseNode`
    # will automatically check if the value is correct and will
    # raise `IncompatibleError` if it is not.
    # (e.g. if `attach_index` is set to "all" and `in_channels`
    # is annotated as `int`, an exception will be raised)
    in_channels: int


    # Specifies the task that this head is used for.
    # When specified, the node is better integrated
    # with the rest of the framework (better error messages,
    # compatibility checking, more powerful automation, etc.).
    task = Tasks.SEGMENTATION

    # Parser to use when converting model to NN Archive
    # to be used with DepthAI.
    parser = "SegmentationParser"

    # `__init__` should always take `**kwargs` as an argument
    # and pass it to the parent constructor.
    @override
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.up1 = UpBlock(self.in_channels, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        self.final = nn.Conv2d(32, self.n_classes, kernel_size=1)

    # The `forward` method is the main entry point for the node.
    # It should take either a single tensor or a list of tensors
    # and return again a single tensor or a list of tensors.
    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.final(x)
```

## Custom Attached Modules

There are 3 types of attached modules that can be created:

- `BaseLoss` - used to define custom loss functions
  - Must define the `forward` method
    - Returns one of the following:
      - A single tensor representing the loss value
      - A tuple with the first element being the main loss value and the second element being a dictionary with sub-losses
        - Sub-losses are only used for logging, they do not affect the optimization process
- `BaseMetric` - used to define custom evaluation metrics
  - Must implement the following methods:
    - `update` - called for each batch, updates the internal state of the metric
    - `compute` - called at the end of the epoch, computes and returns the final metric value
      - Returns one of the following:
        - A single tensor with the result
        - A tuple where the first element is the main result and the second element is a dictionary with additional metrics
          - If this metric is marked as the main metric of the model, then the main result is used for saving best models and/or early stopping
        - A dictionary of metrics
          - If this is the case, then this metric cannot be used as the main metric of the model
    - `reset` - called at the end of the epoch, resets the internal state
- `BaseVisualizer` - used to define custom visualization methods
  - Must define the `forward` method
  - Returns one of the following:
    - A single image as a torch tensor
    - A tuple of two images; visualization of the targets and predictions

The arguments of the `forward` and `update` methods are special and should follow a specific set of rules to make maximum use of the framework's automation capabilities, (see [Automatic Inputs and Labels Extraction](#automatic-inputs-and-labels-extraction)).

Each attached module should define the following class attributes:

- `supported_tasks: Sequence[Task]` - specifies with which tasks the module is compatible
  - Used to check the compatibility of the module with the task of the connected node and to automatically extract the correct values from the node output and dataset (see [Automatic Inputs and Labels Extraction](#automatic-inputs-and-labels-extraction))
- `node` - In case the module is only compatible with a specific node, you can provide a class-level type hint to the `node` attribute. This will check the compatibility of the module with the node at initialization time.

### Automatic Inputs and Labels Extraction

The framework provides a way to automatically extract the correct values from the node output and dataset based on the task that the connected node is used for. This is done by following a set of rules when defining the `forward` (or `update`) method of the module.

**Rules for Automatic Parameters:**

The signature of the `forward` or `update` can use the following special argument names:

- `predictions` - used to extract the main output of the connected node
  - e.g. if the node's `task` is `Tasks.SEGMENTATION`, the `predictions` argument will be extracted from the node output dictionary using the key `Task.Segmentation.main_output` ("segmentation" in this case)
  - `preds`, `pred`, and `prediction` can also be used
- `target` - used to extract the required label from the dataset
  - e.g. if the node's `task` is `Tasks.SEGMENTATION`, the `target` argument will be extracted from the dataset based on the value of `Task.Segmentation.required_labels` (label of type "segmentation" in this case)
  - Can only be used if the task requires only one label type
    - e.g. cannot be used for instance segmentation
  - `targets` can also be used
- `target_{label_type}` - used to extract a specific label from the dataset
  - e.g. `target_segmentation` will extract the label of type "segmentation" from the dataset
- any other argument will be extracted from the node output dictionary based on the argument name
  - e.g. if the argument is named `features`, the value will be extracted from the node output dictionary using the key `"features"`

> [!NOTE]
> If the arguments are annotated (either `Tensor` or `list[Tensor]`), the framework will check if the types are correct and raise an exception if they are not.

> [!IMPORTANT]
> If the argument is annotated as optional and cannot be extracted, its value will be set to `None`.

> [!TIP]
> Need more control? If the automatic extraction doesn't work for your use case, you can override `run` (or `run_update`) method. These methods are called with 2 arguments; the raw output packet from the connected node and the full label dictionary from the dataset. The return type of these methods is equivalent to the return type of the corresponding `forward` or `update`. Note that this is not recommended and should not be necessary in the vast majority of cases.

### Attached Modules Examples

#### Simple Loss

```python

from typing import override

from torch import Tensor, nn

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks


class BCEWithLogitsLoss(BaseLoss):

    # The `supported_tasks` attribute is used to specify
    # which tasks this loss is compatible with.
    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    @override
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    # The `forward` method is the main entry point for the loss.
    # Arguments of the `forward` method are special and used to
    # automatically extract correct values from the output of the
    # connected node and labels from the dataset.
    #
    # For example, if this module would be connected to a node that
    # defines its task as `Tasks.SEGMENTATION`, the `forward` method
    # will look for `Task.Segmentation.main_output` ("segmentation")
    # in the node output dictionary and use it as the first argument.
    # The second argument will be extracted from the dataset
    # based on the value of `Task.Segmentation.required_labels`
    # (label of type "segmentation" in this case).
    @override
    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        return self.criterion(predictions, target)
```

#### Complex Loss

```python
from typing import override

from torch import Tensor

from luxonis_train.attached_modules.losses import BaseLoss
from luxonis_train.tasks import Tasks

# Example node that produces multiple outputs during
# training that are all used in the loss calculation.
class EfficientKeypointBBoxHead(...):
    def wrap(...) -> Packet[Tensor]:
        return {
            "features": features,
            "class_scores": cls_tensor,
            "distributions": reg_tensor,
            "keypoints_raw": kpt_tensor,
        }

class InstanceKeypointsLoss(BaseLoss):

    # If the loss can only be used with a specific node,
    # you can add a class-level type hint to the `node` attribute.
    # This will check the compatibility of the loss with the node
    # at initialization time.
    node: EfficientKeypointBBoxHead

    # The `Tasks.INSTANCE_KEYPOINTS` task defines two
    # required labels: `"boundingbox"` and `"keypoints"`.
    supported_tasks = [Tasks.INSTANCE_KEYPOINTS]

    # This `forward` method requires multiple arguments
    # from the node and additional it requires more than
    # one label from the dataset.
    #
    # To make use of the automatic parameter extraction,
    # the method signature must follow the rules defined
    # in the 3rd rule for automatic parameters.
    @override
    def forward(
        self,
        features: list[Tensor],
        class_scores: Tensor,
        distributions: Tensor,
        keypoints_raw: Tensor,
        target_boundingbox: Tensor,
        target_keypoints: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

```

#### Metric

The rules for defining the `update` method are the same as for the `forward` method of the loss.

**Metric States**

For better integration with distributed training and easier handling of the metric state,
the metric attributes that are used to store the state of the metric should be
registered using the `add_state` method, see the [torchmetrics documentation](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html).
In order for type checking to pass, the attributes defined using `add_state` should be also added as a class-level annotations.
To streamline this process, `LuxonisTrain` offers a simpler way to define the metric state using the `MetricState` class.
The `MetricState` is intended to be used inside an `Annotated` type for class-level declarations of the metric states.

**Example:**

```python

from luxonis_train import BaseMetric, MetricState

class MyMetric(BaseMetric):
    true_positives: Annotated[Tensor, MetricState(default=0)]
    false_positives: Annotated[Tensor, MetricState(default=0)]
    total: Annotated[Tensor, MetricState(default=0)]

```

The `MetricState` takes the same arguments as `add_state` method, but also specifies some sane default values and conversions:

- If `default` is not specified:
  - If the state is a `Tensor`, the default value is `torch.tensor(0, dtype=torch.float32)`
  - If the state is a `list`, the default value is an empty list
- If `dist_reduce_fx` is not specified:
  - If the state is a `Tensor`, the default value is `"sum"`
  - If the state is a `list`, the default value is `"cat"`

#### Visualizer

The rules for defining the `forward` method are the same as for the `forward` method of the loss.
In addition to the standard set of arguments, the `forward` method also always receives `target_canvas` and `prediction_canvas` arguments containing the original image. The visualizer can use these to overlay the predictions and targets on top of the input image.

> [!IMPORTANT]
> The target arguments should be optional in order for the visualizer to work with predictions only.

```python
from typing import override

from torch import Tensor

from luxonis_train.attached_modules.visualizers import BaseVisualizer
from luxonis_train.tasks import Tasks

class BBoxVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.BOUNDINGBOX]

    @override
    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: list[Tensor],
        target: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:

        # Common visualizer pattern

        predictions_viz = draw_predictions(prediction_canvas, predictions)

        if target is None:
            return predictions_viz

        target_viz = draw_targets(target_canvas, target)
        return target_viz, predictions_viz

```

## Testing Custom Components

To help you test your own nodes end‑to‑end, here is an example script that shows how the luxonis-train pipeline works under the hood. It demonstrates, in the same way `luxonis-train` does internally:

- Defining custom `BaseNode` and `BaseHead` classes.

- Running data through backbone and head.

- Computing a loss using `CrossEntropyLoss`.

in the same way as luxonis-train does it internally.

```python
from luxonis_train import Tasks, BaseHead, BaseNode
from luxonis_train.utils.dataset_metadata import DatasetMetadata
from luxonis_train.attached_modules import CrossEntropyLoss

import torch
from torch import nn, Tensor, Size

class XORBackbone(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = nn.Sequential(
            nn.Linear(2, 10), nn.ReLU()
        )

    def unwrap(self, x: dict[str, Tensor]) -> Tensor:
        x = x[0]["features"][0]
        x = x.view(-1, 2)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

class XORHead(BaseHead):
    task = Tasks.CLASSIFICATION
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = nn.Linear(10, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

    def wrap(self, x: Tensor):
         return {
             "classification": x,
         }

# Setup metadata & nodes
original_in_shape = Size([1, 1, 2])
dataset_metadata = DatasetMetadata(
    classes={'': {'xor_0': 0, 'xor_1': 1
    }},
)

backbone_node = XORBackbone(
    input_shapes = [{"features": original_in_shape}],
    original_in_shape = original_in_shape,
    dataset_metadata = dataset_metadata
)

head_node = XORHead(
    input_shapes = [{"features": Size([1, 10])}],
    original_in_shape = original_in_shape,
    dataset_metadata = dataset_metadata
)

loss = CrossEntropyLoss(node=head_node)

# Dummy data
input = [{"features": [Tensor([[[[0, 0]]]])]}]
target = {"/classification": torch.tensor([0])}

# Forward pass
# 1) Backbone
unwraped_input = backbone_node.unwrap(input)
features = backbone_node.forward(unwraped_input)
wrapped_features = [backbone_node.wrap(features)] # list since next node can have multiple inputs (default lxt behavior)

# 2) Head
unwraped_features = head_node.unwrap(wrapped_features)
logits = head_node.forward(unwraped_features)
wrapped_logits = head_node.wrap(logits)

# 3) Loss
loss_data = loss.get_parameters(wrapped_logits, target)
loss_value = loss.forward(loss_data['predictions'], loss_data['target'])

print(f"Loss value: {loss_value.item()}")
```

Once your `LuxonisModel` is defined, you can run a forward pass like this:

```python
model = LuxonisModel(config)
input = {
     "image": Tensor([[[[0, 0]]]]),
}
model_output = model.lightning_module.forward(inputs=input, compute_loss=False, compute_metrics=False, compute_visualizations=False)
```

To also compute the loss, metrics, and visualizations, simply provide the appropriate targets and set those flags to `True`.
