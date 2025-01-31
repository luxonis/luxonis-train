# Extending the Framework

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

The `luxonis-train` framework is designed to be easily extendable. This document describes how to create custom nodes, losses, metrics, and visualizers.

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
If the model is not intended to be used with `DepthAI`, it is possible to use `BaseNode` for heads as well.

To make the most use out of the framework, the nodes should define the following class attributes:

- `attach_index: int | tuple[int, int] | tuple[int, int, int] | Literal["all"]` - specifies which output of the previous node to use
  - Can be either a single integer (negative indexing is supported), a tuple of integers (slice), or the string `"all"`
  - Typically used for heads that are usually connected to backbones producing a list of feature maps
  - If not specified, it is inferred from the type signature of the `forward` method (if possible)
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
  - To define a custom task, see [Custom Tasks](#custom-tasks)

`BaseNode` defines several convenient properties that can be used to access information about the model:

- `in_channels: int | list[int]` - number of input channels
  - The output is either a single integer or a list of integers depending on the value of `attach_index`
    - That is, if the node is attached to a backbone producing a list of feature maps and the value of `attach_index` is set to `"all"`, `in_channels` will be a list of the channel counts of each feature map
  - Works only if the `attach_index` is defined (or could be inferred)
- `in_width: int | list[int]` - width of the input to the node
- `in_height: int | list[int]` - height of the input to the node
- `n_classes: int` - number of classes
- `n_keypoints: int` - number of keypoints (if applicable)
- `class_names: list[str]` - list of class names
- `original_in_shape: torch.Size` - shape of the original input image
  - Useful for segmentation heads that need to upsample the output to the original image size

> \[!TIP\]
> You can add a class-level type hint to `in_channels`, `in_width`, and `in_height`. This will cause the values to be checked at initialization time and an exception will be raised if the annotation is incompatible with the outputs of the preceding node. (_e.g._ setting `attached_index` to `"all"` and annotating `in_channels` as `int` will raise an exception)

The main methods of the node are:

- `__init__` - constructor
  - Should always take `**kwargs` as an argument and pass it to the parent constructor
- `forward` - main entry point for the node
  - Should take either a single tensor or a list of tensors and return again a single tensor or a list of tensors
- `wrap` - called after `forward`, wraps the output of the node into a dictionary
  - The results of `forward` are not the final outputs of the node, but are wrapped into a dictionary (called a `Packet`)
  - The keys of the dictionary are used to extract the correct values in the attached modules
  - Usually needs to be overridden for heads only
  - The default implementation roughly behaves like this:
    - For backbones and necks, the output is wrapped into a dictionary with a single key `"features"`
    - For heads, the output is wrapped into a dictionary with a single key equivalent to the value of `node.task.main_output` property
      - If not defined, the node is considered to be a backbone or a neck (_i.e._ using the `"features"` key)
    - Roughly equivalent to:
      ```python
      def wrap(self, output: ForwardOutputType) -> Packet[Tensor]:
          if self.task is not None:
              return {self.task.main_output: output}
          return {"features": output}
      ```
- `unwrap` - called before `forward`, the output of `unwrap` is passed to the `forward` method
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

- `get_custom_ead_config` - returns a dictionary with custom head configuration
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
from luxonis

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
    # raise `IncompatibleException` if it is not.
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

The signature of the `forward` or `update` method must follow one of these rules to make use of the automatic parameter extraction:

- It has only 2 arguments, first one for predictions and second for targets.
  - The first argument will be taken from the node output based on the value of `Task.main_output` property.
  - The second argument will be taken from the dataset based on the value of `Task.required_labels` property.
  - Works only for tasks that require a single label, _i.e._ wouldn't work for `Task.KEYPOINTS` because it requires both `"boundingbox"` and `"keypoints"` labels
- Single prediction argument named one of `pred`, `preds`, `prediction`, or `predictions`. One or more target arguments.
  - If one target argument, it has to start with `target`
  - If more than one (_e.g._ for `Tasks.KEYPOINTS`) the name should be `target_{name_of_label}`
    - `target_boundingbox`, `target_keypoints`
- Multiple prediction arguments named the same way as keys in the node output (output of the `BaseNode.wrap` method)
  - Same rules for target arguments as in the rule above.

> \[!NOTE\]
> If the arguments are annotated (either `Tensor` or `list[Tensor]`), the framework will check if the types are correct and raise an exception if they are not.

> \[!IMPORTANT\]
> If the argument is annotated as optional and cannot be extracted, its value will be set to `None`.

> \[!TIP\]
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

#### Visualizer

The rules for defining the `forward` method are the same as for the `forward` method of the loss.
In addition to the standard set of arguments, the `forward` method also always receives `target_canvas` and `prediction_canvas` arguments containing the original image. The visualizer can use these to overlay the predictions and targets on top of the input image.

> \[!IMPORTANT\]
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
