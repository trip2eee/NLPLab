݈
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*	2.0.0-rc12v2.0.0-rc0-101-gd2d2566eef8��
�
cbow/embedding/embeddingsVarHandleOp*
shape
:
**
shared_namecbow/embedding/embeddings*
dtype0*
_output_shapes
: 
�
-cbow/embedding/embeddings/Read/ReadVariableOpReadVariableOpcbow/embedding/embeddings*
dtype0*
_output_shapes

:

�
cbow/embedding_1/embeddingsVarHandleOp*
shape
:
*,
shared_namecbow/embedding_1/embeddings*
dtype0*
_output_shapes
: 
�
/cbow/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpcbow/embedding_1/embeddings*
dtype0*
_output_shapes

:

�
cbow/embedding_2/embeddingsVarHandleOp*
shape
:
*,
shared_namecbow/embedding_2/embeddings*
dtype0*
_output_shapes
: 
�
/cbow/embedding_2/embeddings/Read/ReadVariableOpReadVariableOpcbow/embedding_2/embeddings*
dtype0*
_output_shapes

:


NoOpNoOp
�
ConstConst"/device:CPU:0*�
value�B� B�
�
embd_in

embd_out_w
embd_out_bias
dot

activation
trainable_variables
	variables
regularization_losses
		keras_api


signatures
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
 regularization_losses
!	keras_api

0
1
2

0
1
2
 
�

"layers
#metrics
trainable_variables
	variables
$layer_regularization_losses
regularization_losses
%non_trainable_variables
 
\Z
VARIABLE_VALUEcbow/embedding/embeddings-embd_in/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

&layers
'metrics
trainable_variables
	variables
(layer_regularization_losses
regularization_losses
)non_trainable_variables
a_
VARIABLE_VALUEcbow/embedding_1/embeddings0embd_out_w/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

*layers
+metrics
trainable_variables
	variables
,layer_regularization_losses
regularization_losses
-non_trainable_variables
db
VARIABLE_VALUEcbow/embedding_2/embeddings3embd_out_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�

.layers
/metrics
trainable_variables
	variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
 
 
 
�

2layers
3metrics
trainable_variables
	variables
4layer_regularization_losses
regularization_losses
5non_trainable_variables
 
 
 
�

6layers
7metrics
trainable_variables
	variables
8layer_regularization_losses
 regularization_losses
9non_trainable_variables
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
z
serving_default_input_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cbow/embedding/embeddingscbow/embedding_1/embeddingscbow/embedding_2/embeddings*-
_gradient_op_typePartitionedCall-262956*-
f(R&
$__inference_signature_wrapper_262873*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-cbow/embedding/embeddings/Read/ReadVariableOp/cbow/embedding_1/embeddings/Read/ReadVariableOp/cbow/embedding_2/embeddings/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-262981*(
f#R!
__inference__traced_save_262980*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecbow/embedding/embeddingscbow/embedding_1/embeddingscbow/embedding_2/embeddings*-
_gradient_op_typePartitionedCall-263003*+
f&R$
"__inference__traced_restore_263002*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
_output_shapes
: ��
�
�
__inference__traced_save_262980
file_prefix8
4savev2_cbow_embedding_embeddings_read_readvariableop:
6savev2_cbow_embedding_1_embeddings_read_readvariableop:
6savev2_cbow_embedding_2_embeddings_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_a38b5375bdc9443186758b3f6d4fbe1c/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B-embd_in/embeddings/.ATTRIBUTES/VARIABLE_VALUEB0embd_out_w/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3embd_out_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:s
SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_cbow_embedding_embeddings_read_readvariableop6savev2_cbow_embedding_1_embeddings_read_readvariableop6savev2_cbow_embedding_2_embeddings_read_readvariableop"/device:CPU:0*
dtypes
2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*5
_input_shapes$
": :
:
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : :+ '
%
_user_specified_namefile_prefix: : 
�
b
F__inference_activation_layer_call_and_return_conditional_losses_262834

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
G
+__inference_activation_layer_call_fn_262946

inputs
identity�
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-262840*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_262834*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_263002
file_prefix.
*assignvariableop_cbow_embedding_embeddings2
.assignvariableop_1_cbow_embedding_1_embeddings2
.assignvariableop_2_cbow_embedding_2_embeddings

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B-embd_in/embeddings/.ATTRIBUTES/VARIABLE_VALUEB0embd_out_w/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3embd_out_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:v
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2* 
_output_shapes
:::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_cbow_embedding_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_cbow_embedding_1_embeddingsIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_cbow_embedding_2_embeddingsIdentity_2:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: �

Identity_4IdentityIdentity_3:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "!

identity_4Identity_4:output:0*!
_input_shapes
: :::2
	RestoreV2	RestoreV22(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22
RestoreV2_1RestoreV2_12$
AssignVariableOpAssignVariableOp: :+ '
%
_user_specified_namefile_prefix: : 
�
�
$__inference_signature_wrapper_262873
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-262867**
f%R#
!__inference__wrapped_model_262712*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : 
�
i
?__inference_dot_layer_call_and_return_conditional_losses_262813

inputs
inputs_1
identityN
MulMulinputsinputs_1*
T0*'
_output_shapes
:���������W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: a
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: q

ExpandDims
ExpandDimsSum:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:&"
 
_user_specified_nameinputs:& "
 
_user_specified_nameinputs
�
�
%__inference_cbow_layer_call_fn_262861
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-262855*I
fDRB
@__inference_cbow_layer_call_and_return_conditional_losses_262849*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : 
�

�
G__inference_embedding_2_layer_call_and_return_conditional_losses_262914

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�

�
E__inference_embedding_layer_call_and_return_conditional_losses_262735

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*+
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�

�
G__inference_embedding_2_layer_call_and_return_conditional_losses_262787

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�
�
*__inference_embedding_layer_call_fn_262890

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262741*N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_262735*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
�

�
G__inference_embedding_1_layer_call_and_return_conditional_losses_262899

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�&
�
@__inference_cbow_layer_call_and_return_conditional_losses_262849
input_1,
(embedding_statefulpartitionedcall_args_1.
*embedding_1_statefulpartitionedcall_args_1.
*embedding_2_statefulpartitionedcall_args_1
identity��!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCalld
strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:f
strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:f
strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*'
_output_shapes
:���������f
strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:h
strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0(embedding_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262741*N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_262735*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*+
_output_shapes
:����������
embedding/IdentityIdentity*embedding/StatefulPartitionedCall:output:0"^embedding/StatefulPartitionedCall*
T0*+
_output_shapes
:���������X
Mean/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: |
MeanMeanembedding/Identity:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0*embedding_1_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262768*P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_262762*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
embedding_1/IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:����������
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0*embedding_2_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262793*P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_262787*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
embedding_2/IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:����������
dot/PartitionedCallPartitionedCallMean:output:0embedding_1/Identity:output:0*-
_gradient_op_typePartitionedCall-262820*H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_262813*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������h
dot/IdentityIdentitydot/PartitionedCall:output:0*
T0*'
_output_shapes
:���������r
AddAdddot/Identity:output:0embedding_2/Identity:output:0*
T0*'
_output_shapes
:����������
activation/PartitionedCallPartitionedCallAdd:z:0*-
_gradient_op_typePartitionedCall-262840*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_262834*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������v
activation/IdentityIdentity#activation/PartitionedCall:output:0*
T0*'
_output_shapes
:����������
IdentityIdentityactivation/Identity:output:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*2
_input_shapes!
:���������:::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : 
�
�
,__inference_embedding_1_layer_call_fn_262905

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262768*P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_262762*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
�
�
,__inference_embedding_2_layer_call_fn_262920

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-262793*P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_262787*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
�
P
$__inference_dot_layer_call_fn_262936
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*-
_gradient_op_typePartitionedCall-262820*H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_262813*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0
�

�
E__inference_embedding_layer_call_and_return_conditional_losses_262884

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*+
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�

�
G__inference_embedding_1_layer_call_and_return_conditional_losses_262762

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp: :& "
 
_user_specified_nameinputs
�
k
?__inference_dot_layer_call_and_return_conditional_losses_262930
inputs_0
inputs_1
identityP
MulMulinputs_0inputs_1*
T0*'
_output_shapes
:���������W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: a
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: q

ExpandDims
ExpandDimsSum:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������[
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0
�1
�
!__inference__wrapped_model_262712
input_1;
7cbow_embedding_embedding_lookup_readvariableop_resource=
9cbow_embedding_1_embedding_lookup_readvariableop_resource=
9cbow_embedding_2_embedding_lookup_readvariableop_resource
identity��.cbow/embedding/embedding_lookup/ReadVariableOp�0cbow/embedding_1/embedding_lookup/ReadVariableOp�0cbow/embedding_2/embedding_lookup/ReadVariableOpi
cbow/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:k
cbow/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:k
cbow/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
cbow/strided_sliceStridedSliceinput_1!cbow/strided_slice/stack:output:0#cbow/strided_slice/stack_1:output:0#cbow/strided_slice/stack_2:output:0*
T0*
Index0*

begin_mask*
end_mask*'
_output_shapes
:���������k
cbow/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:m
cbow/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:m
cbow/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:�
cbow/strided_slice_1StridedSliceinput_1#cbow/strided_slice_1/stack:output:0%cbow/strided_slice_1/stack_1:output:0%cbow/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:����������
.cbow/embedding/embedding_lookup/ReadVariableOpReadVariableOp7cbow_embedding_embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
$cbow/embedding/embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *A
_class7
53loc:@cbow/embedding/embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
cbow/embedding/embedding_lookupGatherV26cbow/embedding/embedding_lookup/ReadVariableOp:value:0cbow/strided_slice:output:0-cbow/embedding/embedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@cbow/embedding/embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:����������
(cbow/embedding/embedding_lookup/IdentityIdentity(cbow/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������]
cbow/Mean/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
	cbow/MeanMean1cbow/embedding/embedding_lookup/Identity:output:0$cbow/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0cbow/embedding_1/embedding_lookup/ReadVariableOpReadVariableOp9cbow_embedding_1_embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
&cbow/embedding_1/embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *C
_class9
75loc:@cbow/embedding_1/embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
!cbow/embedding_1/embedding_lookupGatherV28cbow/embedding_1/embedding_lookup/ReadVariableOp:value:0cbow/strided_slice_1:output:0/cbow/embedding_1/embedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@cbow/embedding_1/embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:����������
*cbow/embedding_1/embedding_lookup/IdentityIdentity*cbow/embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
0cbow/embedding_2/embedding_lookup/ReadVariableOpReadVariableOp9cbow_embedding_2_embedding_lookup_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
�
&cbow/embedding_2/embedding_lookup/axisConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B : *C
_class9
75loc:@cbow/embedding_2/embedding_lookup/ReadVariableOp*
dtype0*
_output_shapes
: �
!cbow/embedding_2/embedding_lookupGatherV28cbow/embedding_2/embedding_lookup/ReadVariableOp:value:0cbow/strided_slice_1:output:0/cbow/embedding_2/embedding_lookup/axis:output:0",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@cbow/embedding_2/embedding_lookup/ReadVariableOp*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:����������
*cbow/embedding_2/embedding_lookup/IdentityIdentity*cbow/embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
cbow/dot/MulMulcbow/Mean:output:03cbow/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������`
cbow/dot/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: |
cbow/dot/SumSumcbow/dot/Mul:z:0'cbow/dot/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:���������Y
cbow/dot/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: �
cbow/dot/ExpandDims
ExpandDimscbow/dot/Sum:output:0 cbow/dot/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
cbow/AddAddcbow/dot/ExpandDims:output:03cbow/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������b
cbow/activation/SigmoidSigmoidcbow/Add:z:0*
T0*'
_output_shapes
:����������
IdentityIdentitycbow/activation/Sigmoid:y:0/^cbow/embedding/embedding_lookup/ReadVariableOp1^cbow/embedding_1/embedding_lookup/ReadVariableOp1^cbow/embedding_2/embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*2
_input_shapes!
:���������:::2`
.cbow/embedding/embedding_lookup/ReadVariableOp.cbow/embedding/embedding_lookup/ReadVariableOp2d
0cbow/embedding_1/embedding_lookup/ReadVariableOp0cbow/embedding_1/embedding_lookup/ReadVariableOp2d
0cbow/embedding_2/embedding_lookup/ReadVariableOp0cbow/embedding_2/embedding_lookup/ReadVariableOp: :' #
!
_user_specified_name	input_1: : 
�
b
F__inference_activation_layer_call_and_return_conditional_losses_262941

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:���������S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�_
�
embd_in

embd_out_w
embd_out_bias
dot

activation
trainable_variables
	variables
regularization_losses
		keras_api


signatures
*:&call_and_return_all_conditional_losses
;__call__
<_default_save_signature"�
_tf_keras_model�{"class_name": "CBOW", "name": "cbow", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "CBOW"}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 10, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 10, "output_dim": 2, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
�

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 10, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
�
trainable_variables
	variables
regularization_losses
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"�
_tf_keras_layer�{"class_name": "Dot", "name": "dot", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dot", "trainable": true, "dtype": "float32", "axes": [1, 1], "normalize": false}}
�
trainable_variables
	variables
 regularization_losses
!	keras_api
*E&call_and_return_all_conditional_losses
F__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

"layers
#metrics
trainable_variables
	variables
$layer_regularization_losses
regularization_losses
%non_trainable_variables
;__call__
<_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
+:)
2cbow/embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�

&layers
'metrics
trainable_variables
	variables
(layer_regularization_losses
regularization_losses
)non_trainable_variables
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
-:+
2cbow/embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�

*layers
+metrics
trainable_variables
	variables
,layer_regularization_losses
regularization_losses
-non_trainable_variables
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
-:+
2cbow/embedding_2/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�

.layers
/metrics
trainable_variables
	variables
0layer_regularization_losses
regularization_losses
1non_trainable_variables
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

2layers
3metrics
trainable_variables
	variables
4layer_regularization_losses
regularization_losses
5non_trainable_variables
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

6layers
7metrics
trainable_variables
	variables
8layer_regularization_losses
 regularization_losses
9non_trainable_variables
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
@__inference_cbow_layer_call_and_return_conditional_losses_262849�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
%__inference_cbow_layer_call_fn_262861�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
!__inference__wrapped_model_262712�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
E__inference_embedding_layer_call_and_return_conditional_losses_262884�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_layer_call_fn_262890�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_embedding_1_layer_call_and_return_conditional_losses_262899�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_embedding_1_layer_call_fn_262905�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_embedding_2_layer_call_and_return_conditional_losses_262914�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_embedding_2_layer_call_fn_262920�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dot_layer_call_and_return_conditional_losses_262930�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dot_layer_call_fn_262936�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_activation_layer_call_and_return_conditional_losses_262941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_activation_layer_call_fn_262946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_262873input_1�
G__inference_embedding_2_layer_call_and_return_conditional_losses_262914W+�(
!�
�
inputs���������
� "%�"
�
0���������
� z
,__inference_embedding_1_layer_call_fn_262905J+�(
!�
�
inputs���������
� "�����������
@__inference_cbow_layer_call_and_return_conditional_losses_262849^0�-
&�#
!�
input_1���������
� "%�"
�
0���������
� �
F__inference_activation_layer_call_and_return_conditional_losses_262941X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
$__inference_signature_wrapper_262873w;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1����������
!__inference__wrapped_model_262712l0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
?__inference_dot_layer_call_and_return_conditional_losses_262930�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� z
%__inference_cbow_layer_call_fn_262861Q0�-
&�#
!�
input_1���������
� "�����������
*__inference_embedding_layer_call_fn_262890R/�,
%�"
 �
inputs���������
� "�����������
$__inference_dot_layer_call_fn_262936vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
G__inference_embedding_1_layer_call_and_return_conditional_losses_262899W+�(
!�
�
inputs���������
� "%�"
�
0���������
� �
E__inference_embedding_layer_call_and_return_conditional_losses_262884_/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� z
+__inference_activation_layer_call_fn_262946K/�,
%�"
 �
inputs���������
� "����������z
,__inference_embedding_2_layer_call_fn_262920J+�(
!�
�
inputs���������
� "����������