from tensorflow.keras  import backend as K
from tensorflow.keras  import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
import tensorflow as tf

class CGC(Layer):
    # 可能还是使用一个矩阵 然后从中选取出，本任务，接收的Expert的输出
    # 可以使用矩阵乘法完成对应列的筛选
    def __init__(self,
                 units,
                 num_shared_experts,
                 num_tasks,
                 num_task_experts_list=[],
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',

                 # bias
                 expert_bias_initializer='zeros',  # expert
                 gate_bias_initializer='zeros',  # gate
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,

                 # weight
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,

                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.

        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        # Hidden nodes parameter
        ################### add num input ############################
        self.num_shared_experts = num_shared_experts
        self.num_task_experts_list = num_task_experts_list
        num_experts = num_shared_experts + sum(num_task_experts_list)
        ################### end ######################################
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None  # expert
        self.gate_kernels = None  # gate
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = activations.get(expert_activation)
        self.gate_activation = activations.get(gate_activation)

        # Bias parameter
        self.expert_bias = None  # expert
        self.gate_bias = None  # gate
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        super(CGC, self).__init__(**kwargs)

    def build(self, input_shape):
        print("in build")
        """
        Method for creating the layer weights.

        :param input_shape: Keras tensor (future input to layer)
                            or list/tuple of Keras tensors to reference
                            for weight shape computations
        """
        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        # Initialize expert weights (number of input features * number of units per expert * number of experts)
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )
        print("expert_kernels", self.expert_kernels)

        # Initialize expert bias (number of units per expert * number of experts)
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.units, self.num_experts),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            )

        # TODO 1 gate dimension
        # 需要传入各个task专属expert 和 共享expert 个数 mk 和 ms
        # Initialize gate weights (number of input features * number of experts * number of tasks)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # for i in range(self.num_tasks):
        #     print("gate_kernels", self.gate_kernels)
        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        ###################### add gate mask ###############################
        # TODO gate mask
        self.mask_of_gates = []
        expert_start_idx = self.num_shared_experts

        for i in range(self.num_tasks):
            each_mask_of_gate_np = np.zeros((1, self.num_experts))
            each_mask_of_gate_np[0, 0:self.num_shared_experts] = 1  # remain the shared task related experts
            # print("expert_start_idx", expert_start_idx)
            expert_end_idx = expert_start_idx + self.num_task_experts_list[i]
            # print("expert_end_idx", expert_end_idx)
            each_mask_of_gate_np[0, expert_start_idx: expert_end_idx] = 1  # remain the task related experts
            expert_start_idx = expert_end_idx
            print(each_mask_of_gate_np)
            each_mask_of_gate = K.constant(each_mask_of_gate_np)
            # print("each_mask_of_gate", each_mask_of_gate)
            self.mask_of_gates.append(each_mask_of_gate)
        print("self.mask_of_gates", self.mask_of_gates)
        ##################### end add gate mask ###############################
        super(CGC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.

        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)
        print("expert_outputs", expert_outputs)

        # Add the bias term to the expert weights if necessary
        if self.use_expert_bias:
            expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)
            # Add the bias term to the gate weights if necessary
            if self.use_gate_bias:
                gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            ######################## use gate ###########################
            # gate_output = gate_output * mask
            # 这里可以添加掩码 mask 除shared和其专属expert外都为0
            # print("index", index, "gate_output", gate_output, self.mask_of_gates[index])
            gate_output = gate_output * self.mask_of_gates[index]
            # print("gate_output", gate_output)
            ######################## end ################################
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            # print("gate_output_before expand", gate_output)
            expanded_gate_output = K.expand_dims(gate_output, axis=1)
            # print("gate_output_after expand", expanded_gate_output)
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)
            print("weighted_expert_output", weighted_expert_output)
            sum_weighted_expert_output = K.sum(weighted_expert_output, axis=2)
            print("sum_weighted_expert_output", sum_weighted_expert_output)
            final_outputs.append(sum_weighted_expert_output)

        print("final_outputs", final_outputs)
        return final_outputs

    def compute_output_shape(self, input_shape):
        """
        Method for computing the output shape of the MMoE layer.

        :param input_shape: Shape tuple (tuple of integers)
        :return: List of input shape tuple where the size of the list is equal to the number of tasks
        """
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        """
        Method for returning the configuration of the MMoE layer.

        :return: Config dictionary
        """
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'gate_activation': activations.serialize(self.gate_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(CGC, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
if __name__ == "__main__":
    cgc = CGC(units=8, num_shared_experts=4, num_tasks=4, num_task_experts_list=[1, 0, 3, 2])

    input_placeholder = K.placeholder((2, 143))
    output = cgc(input_placeholder)
    # print(output)
