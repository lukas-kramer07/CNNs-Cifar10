ϟ!
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/v/dense_11/bias
y
(Adam/v/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/bias*
_output_shapes
:
*
dtype0
�
Adam/m/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/m/dense_11/bias
y
(Adam/m/dense_11/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/bias*
_output_shapes
:
*
dtype0
�
Adam/v/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*'
shared_nameAdam/v/dense_11/kernel
�
*Adam/v/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_11/kernel*
_output_shapes

: 
*
dtype0
�
Adam/m/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*'
shared_nameAdam/m/dense_11/kernel
�
*Adam/m/dense_11/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_11/kernel*
_output_shapes

: 
*
dtype0
�
"Adam/v/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_17/beta
�
6Adam/v/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_17/beta*
_output_shapes
: *
dtype0
�
"Adam/m/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_17/beta
�
6Adam/m/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_17/beta*
_output_shapes
: *
dtype0
�
#Adam/v/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/v/batch_normalization_17/gamma
�
7Adam/v/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_17/gamma*
_output_shapes
: *
dtype0
�
#Adam/m/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/m/batch_normalization_17/gamma
�
7Adam/m/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_17/gamma*
_output_shapes
: *
dtype0
�
Adam/v/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_10/bias
y
(Adam/v/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_10/bias
y
(Adam/m/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e *'
shared_nameAdam/v/dense_10/kernel
�
*Adam/v/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/kernel*
_output_shapes

:e *
dtype0
�
Adam/m/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e *'
shared_nameAdam/m/dense_10/kernel
�
*Adam/m/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/kernel*
_output_shapes

:e *
dtype0
�
"Adam/v/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*3
shared_name$"Adam/v/batch_normalization_16/beta
�
6Adam/v/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_16/beta*
_output_shapes
:e*
dtype0
�
"Adam/m/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*3
shared_name$"Adam/m/batch_normalization_16/beta
�
6Adam/m/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_16/beta*
_output_shapes
:e*
dtype0
�
#Adam/v/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*4
shared_name%#Adam/v/batch_normalization_16/gamma
�
7Adam/v/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_16/gamma*
_output_shapes
:e*
dtype0
�
#Adam/m/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*4
shared_name%#Adam/m/batch_normalization_16/gamma
�
7Adam/m/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_16/gamma*
_output_shapes
:e*
dtype0
~
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
:e*
dtype0
~
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
:e*
dtype0
�
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�e*&
shared_nameAdam/v/dense_9/kernel
�
)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes
:	�e*
dtype0
�
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�e*&
shared_nameAdam/m/dense_9/kernel
�
)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes
:	�e*
dtype0
�
"Adam/v/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/v/batch_normalization_15/beta
�
6Adam/v/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
"Adam/m/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/m/batch_normalization_15/beta
�
6Adam/m/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
#Adam/v/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/v/batch_normalization_15/gamma
�
7Adam/v/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_15/gamma*
_output_shapes	
:�*
dtype0
�
#Adam/m/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/m/batch_normalization_15/gamma
�
7Adam/m/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_15/gamma*
_output_shapes	
:�*
dtype0

Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_8/bias
x
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_8/bias
x
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_8/kernel
�
)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_8/kernel
�
)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel* 
_output_shapes
:
��*
dtype0
�
"Adam/v/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/v/batch_normalization_14/beta
�
6Adam/v/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_14/beta*
_output_shapes
:8*
dtype0
�
"Adam/m/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/m/batch_normalization_14/beta
�
6Adam/m/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_14/beta*
_output_shapes
:8*
dtype0
�
#Adam/v/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*4
shared_name%#Adam/v/batch_normalization_14/gamma
�
7Adam/v/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_14/gamma*
_output_shapes
:8*
dtype0
�
#Adam/m/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*4
shared_name%#Adam/m/batch_normalization_14/gamma
�
7Adam/m/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_14/gamma*
_output_shapes
:8*
dtype0
�
Adam/v/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*%
shared_nameAdam/v/conv2d_8/bias
y
(Adam/v/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/bias*
_output_shapes
:8*
dtype0
�
Adam/m/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*%
shared_nameAdam/m/conv2d_8/bias
y
(Adam/m/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/bias*
_output_shapes
:8*
dtype0
�
Adam/v/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*'
shared_nameAdam/v/conv2d_8/kernel
�
*Adam/v/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/kernel*&
_output_shapes
:8*
dtype0
�
Adam/m/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*'
shared_nameAdam/m/conv2d_8/kernel
�
*Adam/m/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/kernel*&
_output_shapes
:8*
dtype0
�
"Adam/v/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_13/beta
�
6Adam/v/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_13/beta*
_output_shapes
:*
dtype0
�
"Adam/m/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_13/beta
�
6Adam/m/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_13/beta*
_output_shapes
:*
dtype0
�
#Adam/v/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_13/gamma
�
7Adam/v/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_13/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_13/gamma
�
7Adam/m/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_13/gamma*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_7/bias
y
(Adam/v/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_7/bias
y
(Adam/m/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_7/kernel
�
*Adam/v/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_7/kernel
�
*Adam/m/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/kernel*&
_output_shapes
:*
dtype0
�
"Adam/v/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_12/beta
�
6Adam/v/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_12/beta*
_output_shapes
:*
dtype0
�
"Adam/m/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_12/beta
�
6Adam/m/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_12/beta*
_output_shapes
:*
dtype0
�
#Adam/v/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_12/gamma
�
7Adam/v/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_12/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_12/gamma
�
7Adam/m/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_12/gamma*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_6/kernel
�
*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_6/kernel
�
*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:
*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

: 
*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
: *
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
: *
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
: *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
: *
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:e *
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:e*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:e*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:e*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:e*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:e*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�e*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	�e*
dtype0
�
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_15/moving_variance
�
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_15/moving_mean
�
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_15/beta
�
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_15/gamma
�
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes	
:�*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:�*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
��*
dtype0
�
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*7
shared_name(&batch_normalization_14/moving_variance
�
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
:8*
dtype0
�
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"batch_normalization_14/moving_mean
�
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
:8*
dtype0
�
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*,
shared_namebatch_normalization_14/beta
�
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
:8*
dtype0
�
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*-
shared_namebatch_normalization_14/gamma
�
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
:8*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:8*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:8* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:8*
dtype0
�
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance
�
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean
�
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta
�
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:*
dtype0
�
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma
�
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0
�
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance
�
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean
�
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta
�
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:*
dtype0
�
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma
�
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_3Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_6/kernelconv2d_6/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_8/kerneldense_8/bias&batch_normalization_15/moving_variancebatch_normalization_15/gamma"batch_normalization_15/moving_meanbatch_normalization_15/betadense_9/kerneldense_9/bias&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betadense_10/kerneldense_10/bias&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betadense_11/kerneldense_11/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1388150

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/axis
	0gamma
1beta
2moving_mean
3moving_variance*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_random_generator* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
&0
'1
02
13
24
35
G6
H7
Q8
R9
S10
T11
h12
i13
r14
s15
t16
u17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37*
�
&0
'1
02
13
G4
H5
Q6
R7
h8
i9
r10
s11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
2
�0
�1
�2
�3
�4
�5* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

&0
'1*

&0
'1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
00
11
22
33*

00
11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

G0
H1*

G0
H1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
Q0
R1
S2
T3*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

h0
i1*

h0
i1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
r0
s1
t2
u3*

r0
s1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_10/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_10/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_17/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_17/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_17/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE&batch_normalization_17/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_11/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
`
20
31
S2
T3
t4
u5
�6
�7
�8
�9
�10
�11*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25*
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

20
31*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

t0
u1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/conv2d_6/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_6/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_6/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_6/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_12/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/batch_normalization_12/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/batch_normalization_12/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_12/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_7/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_7/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_7/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_7/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_13/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_13/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_13/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_13/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_8/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_8/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_8/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_8/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_14/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_14/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_14/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_14/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_15/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_15/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_15/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_15/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_16/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_16/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_16/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_16/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_10/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_10/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_10/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_10/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_17/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_17/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_17/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_17/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_11/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_11/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_11/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_11/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_8/kerneldense_8/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancedense_9/kerneldense_9/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_10/kerneldense_10/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_11/kerneldense_11/bias	iterationlearning_rateAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias#Adam/m/batch_normalization_12/gamma#Adam/v/batch_normalization_12/gamma"Adam/m/batch_normalization_12/beta"Adam/v/batch_normalization_12/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias#Adam/m/batch_normalization_13/gamma#Adam/v/batch_normalization_13/gamma"Adam/m/batch_normalization_13/beta"Adam/v/batch_normalization_13/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias#Adam/m/batch_normalization_14/gamma#Adam/v/batch_normalization_14/gamma"Adam/m/batch_normalization_14/beta"Adam/v/batch_normalization_14/betaAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/bias#Adam/m/batch_normalization_15/gamma#Adam/v/batch_normalization_15/gamma"Adam/m/batch_normalization_15/beta"Adam/v/batch_normalization_15/betaAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/bias#Adam/m/batch_normalization_16/gamma#Adam/v/batch_normalization_16/gamma"Adam/m/batch_normalization_16/beta"Adam/v/batch_normalization_16/betaAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/bias#Adam/m/batch_normalization_17/gamma#Adam/v/batch_normalization_17/gamma"Adam/m/batch_normalization_17/beta"Adam/v/batch_normalization_17/betaAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biastotal_1count_1totalcountConst*m
Tinf
d2b*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1389586
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_8/kerneldense_8/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancedense_9/kerneldense_9/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_10/kerneldense_10/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_11/kerneldense_11/bias	iterationlearning_rateAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/bias#Adam/m/batch_normalization_12/gamma#Adam/v/batch_normalization_12/gamma"Adam/m/batch_normalization_12/beta"Adam/v/batch_normalization_12/betaAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/bias#Adam/m/batch_normalization_13/gamma#Adam/v/batch_normalization_13/gamma"Adam/m/batch_normalization_13/beta"Adam/v/batch_normalization_13/betaAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/bias#Adam/m/batch_normalization_14/gamma#Adam/v/batch_normalization_14/gamma"Adam/m/batch_normalization_14/beta"Adam/v/batch_normalization_14/betaAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/bias#Adam/m/batch_normalization_15/gamma#Adam/v/batch_normalization_15/gamma"Adam/m/batch_normalization_15/beta"Adam/v/batch_normalization_15/betaAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/bias#Adam/m/batch_normalization_16/gamma#Adam/v/batch_normalization_16/gamma"Adam/m/batch_normalization_16/beta"Adam/v/batch_normalization_16/betaAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/bias#Adam/m/batch_normalization_17/gamma#Adam/v/batch_normalization_17/gamma"Adam/m/batch_normalization_17/beta"Adam/v/batch_normalization_17/betaAdam/m/dense_11/kernelAdam/v/dense_11/kernelAdam/m/dense_11/biasAdam/v/dense_11/biastotal_1count_1totalcount*l
Tine
c2a*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1389883��
�	
�
8__inference_batch_normalization_15_layer_call_fn_1388591

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387112p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388581:'#
!
_user_specified_name	1388583:'#
!
_user_specified_name	1388585:'#
!
_user_specified_name	1388587
�
�
D__inference_dense_9_layer_call_and_return_conditional_losses_1387513

inputs1
matmul_readvariableop_resource:	�e-
biasadd_readvariableop_resource:e
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�e*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������er
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:e*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������eP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������e�
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�e*
dtype0�
!dense_9/kernel/Regularizer/L2LossL2Loss8dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0*dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_4_1388980L
9dense_9_kernel_regularizer_l2loss_readvariableop_resource:	�e
identity��0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_9_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�e*
dtype0�
!dense_9/kernel/Regularizer/L2LossL2Loss8dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0*dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_9/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: U
NoOpNoOp1^dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�	
�
__inference_loss_fn_3_1388972M
9dense_8_kernel_regularizer_l2loss_readvariableop_resource:
��
identity��0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_8_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!dense_8/kernel/Regularizer/L2LossL2Loss8dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_8/kernel/Regularizer/mulMul)dense_8/kernel/Regularizer/mul/x:output:0*dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: U
NoOpNoOp1^dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�	
�
8__inference_batch_normalization_17_layer_call_fn_1388866

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388856:'#
!
_user_specified_name	1388858:'#
!
_user_specified_name	1388860:'#
!
_user_specified_name	1388862
�
�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388383

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386880

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1388321

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:����������
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_7/kernel/Regularizer/L2LossL2Loss9conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0+conv2d_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1388393

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
,__inference_dropout_10_layer_call_fn_1388275

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387362w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388538

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������8Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������8*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������8i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������8"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
e
,__inference_dropout_14_layer_call_fn_1388794

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�
�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1387421

inputs8
conv2d_readvariableop_resource:8-
biasadd_readvariableop_resource:8
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:8*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8�
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype0�
"conv2d_8/kernel/Regularizer/L2LossL2Loss9conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0+conv2d_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
h
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1388516

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388658

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
e
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388816

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������e[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������e"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�

f
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387448

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������8Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������8*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������8T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������8i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������8"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
�
)__inference_dense_8_layer_call_fn_1388563

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1387471p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388557:'#
!
_user_specified_name	1388559
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387292

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
8__inference_batch_normalization_16_layer_call_fn_1388722

inputs
unknown:e
	unknown_0:e
	unknown_1:e
	unknown_2:e
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388712:'#
!
_user_specified_name	1388714:'#
!
_user_specified_name	1388716:'#
!
_user_specified_name	1388718
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_1388578

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!dense_8/kernel/Regularizer/L2LossL2Loss8dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_8/kernel/Regularizer/mulMul)dense_8/kernel/Regularizer/mul/x:output:0*dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_11_layer_call_and_return_conditional_losses_1387580

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
e
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388685

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388420

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388900

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_dense_10_layer_call_and_return_conditional_losses_1388840

inputs0
matmul_readvariableop_resource:e -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:e *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:e *
dtype0�
"dense_10/kernel/Regularizer/L2LossL2Loss9dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0+dense_10/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������e: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�&
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388638

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
e
,__inference_dropout_12_layer_call_fn_1388521

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387448w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������8<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������822
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388365

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

f
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388811

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������eQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������e*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������eT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������ea
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������e"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�
�
)__inference_dense_9_layer_call_fn_1388694

inputs
unknown:	�e
	unknown_0:e
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1387513o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388688:'#
!
_user_specified_name	1388690
�&
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387112

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_conv2d_8_layer_call_fn_1388429

inputs!
unknown:8
	unknown_0:8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1387421w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������8<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388423:'#
!
_user_specified_name	1388425
�
e
,__inference_dropout_11_layer_call_fn_1388398

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387405w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1388554

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388506

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������8:8:8:8:8:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

f
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387497

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387675

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������8c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������8"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
��
�)
"__inference__wrapped_model_1386862
input_3N
4sequential_2_conv2d_6_conv2d_readvariableop_resource:C
5sequential_2_conv2d_6_biasadd_readvariableop_resource:I
;sequential_2_batch_normalization_12_readvariableop_resource:K
=sequential_2_batch_normalization_12_readvariableop_1_resource:Z
Lsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:\
Nsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_7_conv2d_readvariableop_resource:C
5sequential_2_conv2d_7_biasadd_readvariableop_resource:I
;sequential_2_batch_normalization_13_readvariableop_resource:K
=sequential_2_batch_normalization_13_readvariableop_1_resource:Z
Lsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:\
Nsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_8_conv2d_readvariableop_resource:8C
5sequential_2_conv2d_8_biasadd_readvariableop_resource:8I
;sequential_2_batch_normalization_14_readvariableop_resource:8K
=sequential_2_batch_normalization_14_readvariableop_1_resource:8Z
Lsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:8\
Nsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:8G
3sequential_2_dense_8_matmul_readvariableop_resource:
��C
4sequential_2_dense_8_biasadd_readvariableop_resource:	�T
Esequential_2_batch_normalization_15_batchnorm_readvariableop_resource:	�X
Isequential_2_batch_normalization_15_batchnorm_mul_readvariableop_resource:	�V
Gsequential_2_batch_normalization_15_batchnorm_readvariableop_1_resource:	�V
Gsequential_2_batch_normalization_15_batchnorm_readvariableop_2_resource:	�F
3sequential_2_dense_9_matmul_readvariableop_resource:	�eB
4sequential_2_dense_9_biasadd_readvariableop_resource:eS
Esequential_2_batch_normalization_16_batchnorm_readvariableop_resource:eW
Isequential_2_batch_normalization_16_batchnorm_mul_readvariableop_resource:eU
Gsequential_2_batch_normalization_16_batchnorm_readvariableop_1_resource:eU
Gsequential_2_batch_normalization_16_batchnorm_readvariableop_2_resource:eF
4sequential_2_dense_10_matmul_readvariableop_resource:e C
5sequential_2_dense_10_biasadd_readvariableop_resource: S
Esequential_2_batch_normalization_17_batchnorm_readvariableop_resource: W
Isequential_2_batch_normalization_17_batchnorm_mul_readvariableop_resource: U
Gsequential_2_batch_normalization_17_batchnorm_readvariableop_1_resource: U
Gsequential_2_batch_normalization_17_batchnorm_readvariableop_2_resource: F
4sequential_2_dense_11_matmul_readvariableop_resource: 
C
5sequential_2_dense_11_biasadd_readvariableop_resource:

identity��Csequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp�Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�2sequential_2/batch_normalization_12/ReadVariableOp�4sequential_2/batch_normalization_12/ReadVariableOp_1�Csequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp�Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�2sequential_2/batch_normalization_13/ReadVariableOp�4sequential_2/batch_normalization_13/ReadVariableOp_1�Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp�Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�2sequential_2/batch_normalization_14/ReadVariableOp�4sequential_2/batch_normalization_14/ReadVariableOp_1�<sequential_2/batch_normalization_15/batchnorm/ReadVariableOp�>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_1�>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_2�@sequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOp�<sequential_2/batch_normalization_16/batchnorm/ReadVariableOp�>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_1�>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_2�@sequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOp�<sequential_2/batch_normalization_17/batchnorm/ReadVariableOp�>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_1�>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_2�@sequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOp�,sequential_2/conv2d_6/BiasAdd/ReadVariableOp�+sequential_2/conv2d_6/Conv2D/ReadVariableOp�,sequential_2/conv2d_7/BiasAdd/ReadVariableOp�+sequential_2/conv2d_7/Conv2D/ReadVariableOp�,sequential_2/conv2d_8/BiasAdd/ReadVariableOp�+sequential_2/conv2d_8/Conv2D/ReadVariableOp�,sequential_2/dense_10/BiasAdd/ReadVariableOp�+sequential_2/dense_10/MatMul/ReadVariableOp�,sequential_2/dense_11/BiasAdd/ReadVariableOp�+sequential_2/dense_11/MatMul/ReadVariableOp�+sequential_2/dense_8/BiasAdd/ReadVariableOp�*sequential_2/dense_8/MatMul/ReadVariableOp�+sequential_2/dense_9/BiasAdd/ReadVariableOp�*sequential_2/dense_9/MatMul/ReadVariableOp�
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_6/Conv2DConv2Dinput_33sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:����������
2sequential_2/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_12_readvariableop_resource*
_output_shapes
:*
dtype0�
4sequential_2/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Csequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
4sequential_2/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_6/Relu:activations:0:sequential_2/batch_normalization_12/ReadVariableOp:value:0<sequential_2/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
$sequential_2/max_pooling2d_6/MaxPoolMaxPool8sequential_2/batch_normalization_12/FusedBatchNormV3:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
 sequential_2/dropout_10/IdentityIdentity-sequential_2/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:����������
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_7/Conv2DConv2D)sequential_2/dropout_10/Identity:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:����������
2sequential_2/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_13_readvariableop_resource*
_output_shapes
:*
dtype0�
4sequential_2/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Csequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
4sequential_2/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_7/Relu:activations:0:sequential_2/batch_normalization_13/ReadVariableOp:value:0<sequential_2/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
$sequential_2/max_pooling2d_7/MaxPoolMaxPool8sequential_2/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
 sequential_2/dropout_11/IdentityIdentity-sequential_2/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:����������
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype0�
sequential_2/conv2d_8/Conv2DConv2D)sequential_2/dropout_11/Identity:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8*
paddingVALID*
strides
�
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:8*
dtype0�
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8�
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������8�
2sequential_2/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_14_readvariableop_resource*
_output_shapes
:8*
dtype0�
4sequential_2/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_14_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype0�
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
4sequential_2/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3(sequential_2/conv2d_8/Relu:activations:0:sequential_2/batch_normalization_14/ReadVariableOp:value:0<sequential_2/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������8:8:8:8:8:*
epsilon%o�:*
is_training( �
$sequential_2/max_pooling2d_8/MaxPoolMaxPool8sequential_2/batch_normalization_14/FusedBatchNormV3:y:0*/
_output_shapes
:���������8*
ksize
*
paddingVALID*
strides
�
 sequential_2/dropout_12/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:���������8m
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
sequential_2/flatten_2/ReshapeReshape)sequential_2/dropout_12/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_2/dense_8/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
<sequential_2/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOpEsequential_2_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0x
3sequential_2/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_2/batch_normalization_15/batchnorm/addAddV2Dsequential_2/batch_normalization_15/batchnorm/ReadVariableOp:value:0<sequential_2/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
3sequential_2/batch_normalization_15/batchnorm/RsqrtRsqrt5sequential_2/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes	
:��
@sequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_2_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1sequential_2/batch_normalization_15/batchnorm/mulMul7sequential_2/batch_normalization_15/batchnorm/Rsqrt:y:0Hsequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
3sequential_2/batch_normalization_15/batchnorm/mul_1Mul'sequential_2/dense_8/Relu:activations:05sequential_2/batch_normalization_15/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_2_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
3sequential_2/batch_normalization_15/batchnorm/mul_2MulFsequential_2/batch_normalization_15/batchnorm/ReadVariableOp_1:value:05sequential_2/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_2_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
1sequential_2/batch_normalization_15/batchnorm/subSubFsequential_2/batch_normalization_15/batchnorm/ReadVariableOp_2:value:07sequential_2/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
3sequential_2/batch_normalization_15/batchnorm/add_1AddV27sequential_2/batch_normalization_15/batchnorm/mul_1:z:05sequential_2/batch_normalization_15/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
 sequential_2/dropout_13/IdentityIdentity7sequential_2/batch_normalization_15/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes
:	�e*
dtype0�
sequential_2/dense_9/MatMulMatMul)sequential_2/dropout_13/Identity:output:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������e�
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:e*
dtype0�
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������ez
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������e�
<sequential_2/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOpEsequential_2_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
:e*
dtype0x
3sequential_2/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_2/batch_normalization_16/batchnorm/addAddV2Dsequential_2/batch_normalization_16/batchnorm/ReadVariableOp:value:0<sequential_2/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:e�
3sequential_2/batch_normalization_16/batchnorm/RsqrtRsqrt5sequential_2/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:e�
@sequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_2_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
:e*
dtype0�
1sequential_2/batch_normalization_16/batchnorm/mulMul7sequential_2/batch_normalization_16/batchnorm/Rsqrt:y:0Hsequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:e�
3sequential_2/batch_normalization_16/batchnorm/mul_1Mul'sequential_2/dense_9/Relu:activations:05sequential_2/batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������e�
>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_2_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes
:e*
dtype0�
3sequential_2/batch_normalization_16/batchnorm/mul_2MulFsequential_2/batch_normalization_16/batchnorm/ReadVariableOp_1:value:05sequential_2/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:e�
>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_2_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes
:e*
dtype0�
1sequential_2/batch_normalization_16/batchnorm/subSubFsequential_2/batch_normalization_16/batchnorm/ReadVariableOp_2:value:07sequential_2/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:e�
3sequential_2/batch_normalization_16/batchnorm/add_1AddV27sequential_2/batch_normalization_16/batchnorm/mul_1:z:05sequential_2/batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������e�
 sequential_2/dropout_14/IdentityIdentity7sequential_2/batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������e�
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:e *
dtype0�
sequential_2/dense_10/MatMulMatMul)sequential_2/dropout_14/Identity:output:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� |
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
<sequential_2/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOpEsequential_2_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0x
3sequential_2/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1sequential_2/batch_normalization_17/batchnorm/addAddV2Dsequential_2/batch_normalization_17/batchnorm/ReadVariableOp:value:0<sequential_2/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
3sequential_2/batch_normalization_17/batchnorm/RsqrtRsqrt5sequential_2/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes
: �
@sequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_2_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
1sequential_2/batch_normalization_17/batchnorm/mulMul7sequential_2/batch_normalization_17/batchnorm/Rsqrt:y:0Hsequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: �
3sequential_2/batch_normalization_17/batchnorm/mul_1Mul(sequential_2/dense_10/Relu:activations:05sequential_2/batch_normalization_17/batchnorm/mul:z:0*
T0*'
_output_shapes
:��������� �
>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_2_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0�
3sequential_2/batch_normalization_17/batchnorm/mul_2MulFsequential_2/batch_normalization_17/batchnorm/ReadVariableOp_1:value:05sequential_2/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes
: �
>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_2_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0�
1sequential_2/batch_normalization_17/batchnorm/subSubFsequential_2/batch_normalization_17/batchnorm/ReadVariableOp_2:value:07sequential_2/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
3sequential_2/batch_normalization_17/batchnorm/add_1AddV27sequential_2/batch_normalization_17/batchnorm/mul_1:z:05sequential_2/batch_normalization_17/batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� �
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0�
sequential_2/dense_11/MatMulMatMul7sequential_2/batch_normalization_17/batchnorm/add_1:z:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
sequential_2/dense_11/SoftmaxSoftmax&sequential_2/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������
v
IdentityIdentity'sequential_2/dense_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOpD^sequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_12/ReadVariableOp5^sequential_2/batch_normalization_12/ReadVariableOp_1D^sequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_13/ReadVariableOp5^sequential_2/batch_normalization_13/ReadVariableOp_1D^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_14/ReadVariableOp5^sequential_2/batch_normalization_14/ReadVariableOp_1=^sequential_2/batch_normalization_15/batchnorm/ReadVariableOp?^sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_1?^sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_2A^sequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOp=^sequential_2/batch_normalization_16/batchnorm/ReadVariableOp?^sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_1?^sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_2A^sequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOp=^sequential_2/batch_normalization_17/batchnorm/ReadVariableOp?^sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_1?^sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_2A^sequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOp-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Csequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2�
Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_12/ReadVariableOp2sequential_2/batch_normalization_12/ReadVariableOp2l
4sequential_2/batch_normalization_12/ReadVariableOp_14sequential_2/batch_normalization_12/ReadVariableOp_12�
Csequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2�
Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_13/ReadVariableOp2sequential_2/batch_normalization_13/ReadVariableOp2l
4sequential_2/batch_normalization_13/ReadVariableOp_14sequential_2/batch_normalization_13/ReadVariableOp_12�
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2�
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_14/ReadVariableOp2sequential_2/batch_normalization_14/ReadVariableOp2l
4sequential_2/batch_normalization_14/ReadVariableOp_14sequential_2/batch_normalization_14/ReadVariableOp_12|
<sequential_2/batch_normalization_15/batchnorm/ReadVariableOp<sequential_2/batch_normalization_15/batchnorm/ReadVariableOp2�
>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_1>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_12�
>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_2>sequential_2/batch_normalization_15/batchnorm/ReadVariableOp_22�
@sequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOp@sequential_2/batch_normalization_15/batchnorm/mul/ReadVariableOp2|
<sequential_2/batch_normalization_16/batchnorm/ReadVariableOp<sequential_2/batch_normalization_16/batchnorm/ReadVariableOp2�
>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_1>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_12�
>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_2>sequential_2/batch_normalization_16/batchnorm/ReadVariableOp_22�
@sequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOp@sequential_2/batch_normalization_16/batchnorm/mul/ReadVariableOp2|
<sequential_2/batch_normalization_17/batchnorm/ReadVariableOp<sequential_2/batch_normalization_17/batchnorm/ReadVariableOp2�
>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_1>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_12�
>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_2>sequential_2/batch_normalization_17/batchnorm/ReadVariableOp_22�
@sequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOp@sequential_2/batch_normalization_17/batchnorm/mul/ReadVariableOp2\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource
�
M
1__inference_max_pooling2d_6_layer_call_fn_1388265

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1386929�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
H
,__inference_dropout_10_layer_call_fn_1388280

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387633h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
8__inference_batch_normalization_13_layer_call_fn_1388334

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386952�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388324:'#
!
_user_specified_name	1388326:'#
!
_user_specified_name	1388328:'#
!
_user_specified_name	1388330
��
�
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387762
input_3*
conv2d_6_1387614:
conv2d_6_1387616:,
batch_normalization_12_1387619:,
batch_normalization_12_1387621:,
batch_normalization_12_1387623:,
batch_normalization_12_1387625:*
conv2d_7_1387635:
conv2d_7_1387637:,
batch_normalization_13_1387640:,
batch_normalization_13_1387642:,
batch_normalization_13_1387644:,
batch_normalization_13_1387646:*
conv2d_8_1387656:8
conv2d_8_1387658:8,
batch_normalization_14_1387661:8,
batch_normalization_14_1387663:8,
batch_normalization_14_1387665:8,
batch_normalization_14_1387667:8#
dense_8_1387678:
��
dense_8_1387680:	�-
batch_normalization_15_1387683:	�-
batch_normalization_15_1387685:	�-
batch_normalization_15_1387687:	�-
batch_normalization_15_1387689:	�"
dense_9_1387698:	�e
dense_9_1387700:e,
batch_normalization_16_1387703:e,
batch_normalization_16_1387705:e,
batch_normalization_16_1387707:e,
batch_normalization_16_1387709:e"
dense_10_1387718:e 
dense_10_1387720: ,
batch_normalization_17_1387723: ,
batch_normalization_17_1387725: ,
batch_normalization_17_1387727: ,
batch_normalization_17_1387729: "
dense_11_1387732: 

dense_11_1387734:

identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_7/StatefulPartitionedCall�1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_8/StatefulPartitionedCall�1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp� dense_10/StatefulPartitionedCall�1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�dense_9/StatefulPartitionedCall�0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_6_1387614conv2d_6_1387616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387335�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_12_1387619batch_normalization_12_1387621batch_normalization_12_1387623batch_normalization_12_1387625*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386898�
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1386929�
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387633�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0conv2d_7_1387635conv2d_7_1387637*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1387378�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_13_1387640batch_normalization_13_1387642batch_normalization_13_1387644batch_normalization_13_1387646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386970�
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1387001�
dropout_11/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387654�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0conv2d_8_1387656conv2d_8_1387658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1387421�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_14_1387661batch_normalization_14_1387663batch_normalization_14_1387665batch_normalization_14_1387667*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387042�
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1387073�
dropout_12/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387675�
flatten_2/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1387455�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_8_1387678dense_8_1387680*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1387471�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_15_1387683batch_normalization_15_1387685batch_normalization_15_1387687batch_normalization_15_1387689*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387132�
dropout_13/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387696�
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_9_1387698dense_9_1387700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1387513�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_16_1387703batch_normalization_16_1387705batch_normalization_16_1387707batch_normalization_16_1387709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387212�
dropout_14/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387716�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_10_1387718dense_10_1387720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1387555�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_17_1387723batch_normalization_17_1387725batch_normalization_17_1387727batch_normalization_17_1387729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387292�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_11_1387732dense_11_1387734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_1387580�
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1387614*&
_output_shapes
:*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_7_1387635*&
_output_shapes
:*
dtype0�
"conv2d_7/kernel/Regularizer/L2LossL2Loss9conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0+conv2d_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_8_1387656*&
_output_shapes
:8*
dtype0�
"conv2d_8/kernel/Regularizer/L2LossL2Loss9conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0+conv2d_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1387678* 
_output_shapes
:
��*
dtype0�
!dense_8/kernel/Regularizer/L2LossL2Loss8dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_8/kernel/Regularizer/mulMul)dense_8/kernel/Regularizer/mul/x:output:0*dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1387698*
_output_shapes
:	�e*
dtype0�
!dense_9/kernel/Regularizer/L2LossL2Loss8dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0*dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_10_1387718*
_output_shapes

:e *
dtype0�
"dense_10/kernel/Regularizer/L2LossL2Loss9dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0+dense_10/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall1^dense_8/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2d
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:'#
!
_user_specified_name	1387614:'#
!
_user_specified_name	1387616:'#
!
_user_specified_name	1387619:'#
!
_user_specified_name	1387621:'#
!
_user_specified_name	1387623:'#
!
_user_specified_name	1387625:'#
!
_user_specified_name	1387635:'#
!
_user_specified_name	1387637:'	#
!
_user_specified_name	1387640:'
#
!
_user_specified_name	1387642:'#
!
_user_specified_name	1387644:'#
!
_user_specified_name	1387646:'#
!
_user_specified_name	1387656:'#
!
_user_specified_name	1387658:'#
!
_user_specified_name	1387661:'#
!
_user_specified_name	1387663:'#
!
_user_specified_name	1387665:'#
!
_user_specified_name	1387667:'#
!
_user_specified_name	1387678:'#
!
_user_specified_name	1387680:'#
!
_user_specified_name	1387683:'#
!
_user_specified_name	1387685:'#
!
_user_specified_name	1387687:'#
!
_user_specified_name	1387689:'#
!
_user_specified_name	1387698:'#
!
_user_specified_name	1387700:'#
!
_user_specified_name	1387703:'#
!
_user_specified_name	1387705:'#
!
_user_specified_name	1387707:'#
!
_user_specified_name	1387709:'#
!
_user_specified_name	1387718:' #
!
_user_specified_name	1387720:'!#
!
_user_specified_name	1387723:'"#
!
_user_specified_name	1387725:'##
!
_user_specified_name	1387727:'$#
!
_user_specified_name	1387729:'%#
!
_user_specified_name	1387732:'&#
!
_user_specified_name	1387734
�

�
8__inference_batch_normalization_12_layer_call_fn_1388224

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386898�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388214:'#
!
_user_specified_name	1388216:'#
!
_user_specified_name	1388218:'#
!
_user_specified_name	1388220
�
�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387042

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������8:8:8:8:8:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1387378

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:����������
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_7/kernel/Regularizer/L2LossL2Loss9conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0+conv2d_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
8__inference_batch_normalization_15_layer_call_fn_1388604

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387132p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388594:'#
!
_user_specified_name	1388596:'#
!
_user_specified_name	1388598:'#
!
_user_specified_name	1388600
�
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388920

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
H
,__inference_dropout_12_layer_call_fn_1388526

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387675h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������8"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
e
,__inference_dropout_13_layer_call_fn_1388663

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387497p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_1387455

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
H
,__inference_dropout_11_layer_call_fn_1388403

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387654h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387654

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1388198

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:����������
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�&
�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387272

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: �
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:��������� h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:��������� b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_5_1388988L
:dense_10_kernel_regularizer_l2loss_readvariableop_resource:e 
identity��1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_10_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:e *
dtype0�
"dense_10/kernel/Regularizer/L2LossL2Loss9dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0+dense_10/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_10/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^dense_10/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�

f
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387539

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������eQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������e*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������eT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������ea
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������e"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�
�
.__inference_sequential_2_layer_call_fn_1387924
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:8

unknown_12:8

unknown_13:8

unknown_14:8

unknown_15:8

unknown_16:8

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�e

unknown_24:e

unknown_25:e

unknown_26:e

unknown_27:e

unknown_28:e

unknown_29:e 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 


unknown_36:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:'#
!
_user_specified_name	1387846:'#
!
_user_specified_name	1387848:'#
!
_user_specified_name	1387850:'#
!
_user_specified_name	1387852:'#
!
_user_specified_name	1387854:'#
!
_user_specified_name	1387856:'#
!
_user_specified_name	1387858:'#
!
_user_specified_name	1387860:'	#
!
_user_specified_name	1387862:'
#
!
_user_specified_name	1387864:'#
!
_user_specified_name	1387866:'#
!
_user_specified_name	1387868:'#
!
_user_specified_name	1387870:'#
!
_user_specified_name	1387872:'#
!
_user_specified_name	1387874:'#
!
_user_specified_name	1387876:'#
!
_user_specified_name	1387878:'#
!
_user_specified_name	1387880:'#
!
_user_specified_name	1387882:'#
!
_user_specified_name	1387884:'#
!
_user_specified_name	1387886:'#
!
_user_specified_name	1387888:'#
!
_user_specified_name	1387890:'#
!
_user_specified_name	1387892:'#
!
_user_specified_name	1387894:'#
!
_user_specified_name	1387896:'#
!
_user_specified_name	1387898:'#
!
_user_specified_name	1387900:'#
!
_user_specified_name	1387902:'#
!
_user_specified_name	1387904:'#
!
_user_specified_name	1387906:' #
!
_user_specified_name	1387908:'!#
!
_user_specified_name	1387910:'"#
!
_user_specified_name	1387912:'##
!
_user_specified_name	1387914:'$#
!
_user_specified_name	1387916:'%#
!
_user_specified_name	1387918:'&#
!
_user_specified_name	1387920
�
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387633

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388415

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
8__inference_batch_normalization_14_layer_call_fn_1388457

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387024�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388447:'#
!
_user_specified_name	1388449:'#
!
_user_specified_name	1388451:'#
!
_user_specified_name	1388453
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388789

inputs/
!batchnorm_readvariableop_resource:e3
%batchnorm_mul_readvariableop_resource:e1
#batchnorm_readvariableop_1_resource:e1
#batchnorm_readvariableop_2_resource:e
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:e*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:eP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:e~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:e*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������ez
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:e*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:e*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������eb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386970

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
__inference_loss_fn_2_1388964T
:conv2d_8_kernel_regularizer_l2loss_readvariableop_resource:8
identity��1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp�
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_8_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:8*
dtype0�
"conv2d_8/kernel/Regularizer/L2LossL2Loss9conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0+conv2d_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_8/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
��
�[
 __inference__traced_save_1389586
file_prefix@
&read_disablecopyonread_conv2d_6_kernel:4
&read_1_disablecopyonread_conv2d_6_bias:C
5read_2_disablecopyonread_batch_normalization_12_gamma:B
4read_3_disablecopyonread_batch_normalization_12_beta:I
;read_4_disablecopyonread_batch_normalization_12_moving_mean:M
?read_5_disablecopyonread_batch_normalization_12_moving_variance:B
(read_6_disablecopyonread_conv2d_7_kernel:4
&read_7_disablecopyonread_conv2d_7_bias:C
5read_8_disablecopyonread_batch_normalization_13_gamma:B
4read_9_disablecopyonread_batch_normalization_13_beta:J
<read_10_disablecopyonread_batch_normalization_13_moving_mean:N
@read_11_disablecopyonread_batch_normalization_13_moving_variance:C
)read_12_disablecopyonread_conv2d_8_kernel:85
'read_13_disablecopyonread_conv2d_8_bias:8D
6read_14_disablecopyonread_batch_normalization_14_gamma:8C
5read_15_disablecopyonread_batch_normalization_14_beta:8J
<read_16_disablecopyonread_batch_normalization_14_moving_mean:8N
@read_17_disablecopyonread_batch_normalization_14_moving_variance:8<
(read_18_disablecopyonread_dense_8_kernel:
��5
&read_19_disablecopyonread_dense_8_bias:	�E
6read_20_disablecopyonread_batch_normalization_15_gamma:	�D
5read_21_disablecopyonread_batch_normalization_15_beta:	�K
<read_22_disablecopyonread_batch_normalization_15_moving_mean:	�O
@read_23_disablecopyonread_batch_normalization_15_moving_variance:	�;
(read_24_disablecopyonread_dense_9_kernel:	�e4
&read_25_disablecopyonread_dense_9_bias:eD
6read_26_disablecopyonread_batch_normalization_16_gamma:eC
5read_27_disablecopyonread_batch_normalization_16_beta:eJ
<read_28_disablecopyonread_batch_normalization_16_moving_mean:eN
@read_29_disablecopyonread_batch_normalization_16_moving_variance:e;
)read_30_disablecopyonread_dense_10_kernel:e 5
'read_31_disablecopyonread_dense_10_bias: D
6read_32_disablecopyonread_batch_normalization_17_gamma: C
5read_33_disablecopyonread_batch_normalization_17_beta: J
<read_34_disablecopyonread_batch_normalization_17_moving_mean: N
@read_35_disablecopyonread_batch_normalization_17_moving_variance: ;
)read_36_disablecopyonread_dense_11_kernel: 
5
'read_37_disablecopyonread_dense_11_bias:
-
#read_38_disablecopyonread_iteration:	 1
'read_39_disablecopyonread_learning_rate: J
0read_40_disablecopyonread_adam_m_conv2d_6_kernel:J
0read_41_disablecopyonread_adam_v_conv2d_6_kernel:<
.read_42_disablecopyonread_adam_m_conv2d_6_bias:<
.read_43_disablecopyonread_adam_v_conv2d_6_bias:K
=read_44_disablecopyonread_adam_m_batch_normalization_12_gamma:K
=read_45_disablecopyonread_adam_v_batch_normalization_12_gamma:J
<read_46_disablecopyonread_adam_m_batch_normalization_12_beta:J
<read_47_disablecopyonread_adam_v_batch_normalization_12_beta:J
0read_48_disablecopyonread_adam_m_conv2d_7_kernel:J
0read_49_disablecopyonread_adam_v_conv2d_7_kernel:<
.read_50_disablecopyonread_adam_m_conv2d_7_bias:<
.read_51_disablecopyonread_adam_v_conv2d_7_bias:K
=read_52_disablecopyonread_adam_m_batch_normalization_13_gamma:K
=read_53_disablecopyonread_adam_v_batch_normalization_13_gamma:J
<read_54_disablecopyonread_adam_m_batch_normalization_13_beta:J
<read_55_disablecopyonread_adam_v_batch_normalization_13_beta:J
0read_56_disablecopyonread_adam_m_conv2d_8_kernel:8J
0read_57_disablecopyonread_adam_v_conv2d_8_kernel:8<
.read_58_disablecopyonread_adam_m_conv2d_8_bias:8<
.read_59_disablecopyonread_adam_v_conv2d_8_bias:8K
=read_60_disablecopyonread_adam_m_batch_normalization_14_gamma:8K
=read_61_disablecopyonread_adam_v_batch_normalization_14_gamma:8J
<read_62_disablecopyonread_adam_m_batch_normalization_14_beta:8J
<read_63_disablecopyonread_adam_v_batch_normalization_14_beta:8C
/read_64_disablecopyonread_adam_m_dense_8_kernel:
��C
/read_65_disablecopyonread_adam_v_dense_8_kernel:
��<
-read_66_disablecopyonread_adam_m_dense_8_bias:	�<
-read_67_disablecopyonread_adam_v_dense_8_bias:	�L
=read_68_disablecopyonread_adam_m_batch_normalization_15_gamma:	�L
=read_69_disablecopyonread_adam_v_batch_normalization_15_gamma:	�K
<read_70_disablecopyonread_adam_m_batch_normalization_15_beta:	�K
<read_71_disablecopyonread_adam_v_batch_normalization_15_beta:	�B
/read_72_disablecopyonread_adam_m_dense_9_kernel:	�eB
/read_73_disablecopyonread_adam_v_dense_9_kernel:	�e;
-read_74_disablecopyonread_adam_m_dense_9_bias:e;
-read_75_disablecopyonread_adam_v_dense_9_bias:eK
=read_76_disablecopyonread_adam_m_batch_normalization_16_gamma:eK
=read_77_disablecopyonread_adam_v_batch_normalization_16_gamma:eJ
<read_78_disablecopyonread_adam_m_batch_normalization_16_beta:eJ
<read_79_disablecopyonread_adam_v_batch_normalization_16_beta:eB
0read_80_disablecopyonread_adam_m_dense_10_kernel:e B
0read_81_disablecopyonread_adam_v_dense_10_kernel:e <
.read_82_disablecopyonread_adam_m_dense_10_bias: <
.read_83_disablecopyonread_adam_v_dense_10_bias: K
=read_84_disablecopyonread_adam_m_batch_normalization_17_gamma: K
=read_85_disablecopyonread_adam_v_batch_normalization_17_gamma: J
<read_86_disablecopyonread_adam_m_batch_normalization_17_beta: J
<read_87_disablecopyonread_adam_v_batch_normalization_17_beta: B
0read_88_disablecopyonread_adam_m_dense_11_kernel: 
B
0read_89_disablecopyonread_adam_v_dense_11_kernel: 
<
.read_90_disablecopyonread_adam_m_dense_11_bias:
<
.read_91_disablecopyonread_adam_v_dense_11_bias:
+
!read_92_disablecopyonread_total_1: +
!read_93_disablecopyonread_count_1: )
read_94_disablecopyonread_total: )
read_95_disablecopyonread_count: 
savev2_const
identity_193��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_6_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_batch_normalization_12_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_batch_normalization_12_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_batch_normalization_12_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_batch_normalization_12_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead?read_5_disablecopyonread_batch_normalization_12_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp?read_5_disablecopyonread_batch_normalization_12_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_7_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_7_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead5read_8_disablecopyonread_batch_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp5read_8_disablecopyonread_batch_normalization_13_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_13_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_batch_normalization_13_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_batch_normalization_13_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead@read_11_disablecopyonread_batch_normalization_13_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp@read_11_disablecopyonread_batch_normalization_13_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv2d_8_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:8*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:8m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:8|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv2d_8_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_14/DisableCopyOnReadDisableCopyOnRead6read_14_disablecopyonread_batch_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp6read_14_disablecopyonread_batch_normalization_14_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_15/DisableCopyOnReadDisableCopyOnRead5read_15_disablecopyonread_batch_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp5read_15_disablecopyonread_batch_normalization_14_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_batch_normalization_14_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_batch_normalization_14_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_17/DisableCopyOnReadDisableCopyOnRead@read_17_disablecopyonread_batch_normalization_14_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp@read_17_disablecopyonread_batch_normalization_14_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:8}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_dense_8_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_dense_8_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead6read_20_disablecopyonread_batch_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp6read_20_disablecopyonread_batch_normalization_15_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead5read_21_disablecopyonread_batch_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp5read_21_disablecopyonread_batch_normalization_15_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead<read_22_disablecopyonread_batch_normalization_15_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp<read_22_disablecopyonread_batch_normalization_15_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead@read_23_disablecopyonread_batch_normalization_15_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp@read_23_disablecopyonread_batch_normalization_15_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_24/DisableCopyOnReadDisableCopyOnRead(read_24_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp(read_24_disablecopyonread_dense_9_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�e*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�ef
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�e{
Read_25/DisableCopyOnReadDisableCopyOnRead&read_25_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp&read_25_disablecopyonread_dense_9_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ea
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_26/DisableCopyOnReadDisableCopyOnRead6read_26_disablecopyonread_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp6read_26_disablecopyonread_batch_normalization_16_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ea
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_27/DisableCopyOnReadDisableCopyOnRead5read_27_disablecopyonread_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp5read_27_disablecopyonread_batch_normalization_16_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ea
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_28/DisableCopyOnReadDisableCopyOnRead<read_28_disablecopyonread_batch_normalization_16_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp<read_28_disablecopyonread_batch_normalization_16_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ea
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_29/DisableCopyOnReadDisableCopyOnRead@read_29_disablecopyonread_batch_normalization_16_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp@read_29_disablecopyonread_batch_normalization_16_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ea
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:e~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_dense_10_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:e *
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:e |
Read_31/DisableCopyOnReadDisableCopyOnRead'read_31_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp'read_31_disablecopyonread_dense_10_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead6read_32_disablecopyonread_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp6read_32_disablecopyonread_batch_normalization_17_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead5read_33_disablecopyonread_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp5read_33_disablecopyonread_batch_normalization_17_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_34/DisableCopyOnReadDisableCopyOnRead<read_34_disablecopyonread_batch_normalization_17_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp<read_34_disablecopyonread_batch_normalization_17_moving_mean^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_35/DisableCopyOnReadDisableCopyOnRead@read_35_disablecopyonread_batch_normalization_17_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp@read_35_disablecopyonread_batch_normalization_17_moving_variance^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_dense_11_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 
*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 
e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

: 
|
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_dense_11_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:
x
Read_38/DisableCopyOnReadDisableCopyOnRead#read_38_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp#read_38_disablecopyonread_iteration^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_39/DisableCopyOnReadDisableCopyOnRead'read_39_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp'read_39_disablecopyonread_learning_rate^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_40/DisableCopyOnReadDisableCopyOnRead0read_40_disablecopyonread_adam_m_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp0read_40_disablecopyonread_adam_m_conv2d_6_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_v_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_v_conv2d_6_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_m_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_m_conv2d_6_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead.read_43_disablecopyonread_adam_v_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp.read_43_disablecopyonread_adam_v_conv2d_6_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead=read_44_disablecopyonread_adam_m_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp=read_44_disablecopyonread_adam_m_batch_normalization_12_gamma^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead=read_45_disablecopyonread_adam_v_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp=read_45_disablecopyonread_adam_v_batch_normalization_12_gamma^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead<read_46_disablecopyonread_adam_m_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp<read_46_disablecopyonread_adam_m_batch_normalization_12_beta^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead<read_47_disablecopyonread_adam_v_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp<read_47_disablecopyonread_adam_v_batch_normalization_12_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_conv2d_7_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_conv2d_7_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_conv2d_7_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_conv2d_7_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead=read_52_disablecopyonread_adam_m_batch_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp=read_52_disablecopyonread_adam_m_batch_normalization_13_gamma^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_53/DisableCopyOnReadDisableCopyOnRead=read_53_disablecopyonread_adam_v_batch_normalization_13_gamma"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp=read_53_disablecopyonread_adam_v_batch_normalization_13_gamma^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_54/DisableCopyOnReadDisableCopyOnRead<read_54_disablecopyonread_adam_m_batch_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp<read_54_disablecopyonread_adam_m_batch_normalization_13_beta^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead<read_55_disablecopyonread_adam_v_batch_normalization_13_beta"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp<read_55_disablecopyonread_adam_v_batch_normalization_13_beta^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnRead0read_56_disablecopyonread_adam_m_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp0read_56_disablecopyonread_adam_m_conv2d_8_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:8*
dtype0x
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:8o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
:8�
Read_57/DisableCopyOnReadDisableCopyOnRead0read_57_disablecopyonread_adam_v_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp0read_57_disablecopyonread_adam_v_conv2d_8_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:8*
dtype0x
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:8o
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*&
_output_shapes
:8�
Read_58/DisableCopyOnReadDisableCopyOnRead.read_58_disablecopyonread_adam_m_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp.read_58_disablecopyonread_adam_m_conv2d_8_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_59/DisableCopyOnReadDisableCopyOnRead.read_59_disablecopyonread_adam_v_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp.read_59_disablecopyonread_adam_v_conv2d_8_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_60/DisableCopyOnReadDisableCopyOnRead=read_60_disablecopyonread_adam_m_batch_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp=read_60_disablecopyonread_adam_m_batch_normalization_14_gamma^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_61/DisableCopyOnReadDisableCopyOnRead=read_61_disablecopyonread_adam_v_batch_normalization_14_gamma"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp=read_61_disablecopyonread_adam_v_batch_normalization_14_gamma^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_62/DisableCopyOnReadDisableCopyOnRead<read_62_disablecopyonread_adam_m_batch_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp<read_62_disablecopyonread_adam_m_batch_normalization_14_beta^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_63/DisableCopyOnReadDisableCopyOnRead<read_63_disablecopyonread_adam_v_batch_normalization_14_beta"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp<read_63_disablecopyonread_adam_v_batch_normalization_14_beta^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:8*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:8c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:8�
Read_64/DisableCopyOnReadDisableCopyOnRead/read_64_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp/read_64_disablecopyonread_adam_m_dense_8_kernel^Read_64/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_65/DisableCopyOnReadDisableCopyOnRead/read_65_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp/read_65_disablecopyonread_adam_v_dense_8_kernel^Read_65/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_66/DisableCopyOnReadDisableCopyOnRead-read_66_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp-read_66_disablecopyonread_adam_m_dense_8_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_67/DisableCopyOnReadDisableCopyOnRead-read_67_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp-read_67_disablecopyonread_adam_v_dense_8_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_68/DisableCopyOnReadDisableCopyOnRead=read_68_disablecopyonread_adam_m_batch_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp=read_68_disablecopyonread_adam_m_batch_normalization_15_gamma^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_69/DisableCopyOnReadDisableCopyOnRead=read_69_disablecopyonread_adam_v_batch_normalization_15_gamma"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp=read_69_disablecopyonread_adam_v_batch_normalization_15_gamma^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_70/DisableCopyOnReadDisableCopyOnRead<read_70_disablecopyonread_adam_m_batch_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp<read_70_disablecopyonread_adam_m_batch_normalization_15_beta^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_71/DisableCopyOnReadDisableCopyOnRead<read_71_disablecopyonread_adam_v_batch_normalization_15_beta"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp<read_71_disablecopyonread_adam_v_batch_normalization_15_beta^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_72/DisableCopyOnReadDisableCopyOnRead/read_72_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp/read_72_disablecopyonread_adam_m_dense_9_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�e*
dtype0q
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�eh
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:	�e�
Read_73/DisableCopyOnReadDisableCopyOnRead/read_73_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp/read_73_disablecopyonread_adam_v_dense_9_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�e*
dtype0q
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�eh
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:	�e�
Read_74/DisableCopyOnReadDisableCopyOnRead-read_74_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp-read_74_disablecopyonread_adam_m_dense_9_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_75/DisableCopyOnReadDisableCopyOnRead-read_75_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp-read_75_disablecopyonread_adam_v_dense_9_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_76/DisableCopyOnReadDisableCopyOnRead=read_76_disablecopyonread_adam_m_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp=read_76_disablecopyonread_adam_m_batch_normalization_16_gamma^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_77/DisableCopyOnReadDisableCopyOnRead=read_77_disablecopyonread_adam_v_batch_normalization_16_gamma"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp=read_77_disablecopyonread_adam_v_batch_normalization_16_gamma^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_78/DisableCopyOnReadDisableCopyOnRead<read_78_disablecopyonread_adam_m_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp<read_78_disablecopyonread_adam_m_batch_normalization_16_beta^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_79/DisableCopyOnReadDisableCopyOnRead<read_79_disablecopyonread_adam_v_batch_normalization_16_beta"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp<read_79_disablecopyonread_adam_v_batch_normalization_16_beta^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:e*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ec
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:e�
Read_80/DisableCopyOnReadDisableCopyOnRead0read_80_disablecopyonread_adam_m_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp0read_80_disablecopyonread_adam_m_dense_10_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:e *
dtype0p
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e g
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes

:e �
Read_81/DisableCopyOnReadDisableCopyOnRead0read_81_disablecopyonread_adam_v_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp0read_81_disablecopyonread_adam_v_dense_10_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:e *
dtype0p
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e g
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes

:e �
Read_82/DisableCopyOnReadDisableCopyOnRead.read_82_disablecopyonread_adam_m_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp.read_82_disablecopyonread_adam_m_dense_10_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_83/DisableCopyOnReadDisableCopyOnRead.read_83_disablecopyonread_adam_v_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp.read_83_disablecopyonread_adam_v_dense_10_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_84/DisableCopyOnReadDisableCopyOnRead=read_84_disablecopyonread_adam_m_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp=read_84_disablecopyonread_adam_m_batch_normalization_17_gamma^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_85/DisableCopyOnReadDisableCopyOnRead=read_85_disablecopyonread_adam_v_batch_normalization_17_gamma"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp=read_85_disablecopyonread_adam_v_batch_normalization_17_gamma^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_86/DisableCopyOnReadDisableCopyOnRead<read_86_disablecopyonread_adam_m_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp<read_86_disablecopyonread_adam_m_batch_normalization_17_beta^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_87/DisableCopyOnReadDisableCopyOnRead<read_87_disablecopyonread_adam_v_batch_normalization_17_beta"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp<read_87_disablecopyonread_adam_v_batch_normalization_17_beta^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_88/DisableCopyOnReadDisableCopyOnRead0read_88_disablecopyonread_adam_m_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp0read_88_disablecopyonread_adam_m_dense_11_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 
*
dtype0p
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 
g
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes

: 
�
Read_89/DisableCopyOnReadDisableCopyOnRead0read_89_disablecopyonread_adam_v_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp0read_89_disablecopyonread_adam_v_dense_11_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: 
*
dtype0p
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: 
g
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes

: 
�
Read_90/DisableCopyOnReadDisableCopyOnRead.read_90_disablecopyonread_adam_m_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp.read_90_disablecopyonread_adam_m_dense_11_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_91/DisableCopyOnReadDisableCopyOnRead.read_91_disablecopyonread_adam_v_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp.read_91_disablecopyonread_adam_v_dense_11_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_92/DisableCopyOnReadDisableCopyOnRead!read_92_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp!read_92_disablecopyonread_total_1^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_93/DisableCopyOnReadDisableCopyOnRead!read_93_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp!read_93_disablecopyonread_count_1^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_94/DisableCopyOnReadDisableCopyOnReadread_94_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOpread_94_disablecopyonread_total^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_95/DisableCopyOnReadDisableCopyOnReadread_95_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOpread_95_disablecopyonread_count^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
: �)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�)
value�(B�(aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�
value�B�aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *o
dtypese
c2a	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_192Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_193IdentityIdentity_192:output:0^NoOp*
T0*
_output_shapes
: �'
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp*
_output_shapes
 "%
identity_193Identity_193:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:<8
6
_user_specified_namebatch_normalization_12/gamma:;7
5
_user_specified_namebatch_normalization_12/beta:B>
<
_user_specified_name$"batch_normalization_12/moving_mean:FB
@
_user_specified_name(&batch_normalization_12/moving_variance:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:<	8
6
_user_specified_namebatch_normalization_13/gamma:;
7
5
_user_specified_namebatch_normalization_13/beta:B>
<
_user_specified_name$"batch_normalization_13/moving_mean:FB
@
_user_specified_name(&batch_normalization_13/moving_variance:/+
)
_user_specified_nameconv2d_8/kernel:-)
'
_user_specified_nameconv2d_8/bias:<8
6
_user_specified_namebatch_normalization_14/gamma:;7
5
_user_specified_namebatch_normalization_14/beta:B>
<
_user_specified_name$"batch_normalization_14/moving_mean:FB
@
_user_specified_name(&batch_normalization_14/moving_variance:.*
(
_user_specified_namedense_8/kernel:,(
&
_user_specified_namedense_8/bias:<8
6
_user_specified_namebatch_normalization_15/gamma:;7
5
_user_specified_namebatch_normalization_15/beta:B>
<
_user_specified_name$"batch_normalization_15/moving_mean:FB
@
_user_specified_name(&batch_normalization_15/moving_variance:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_9/bias:<8
6
_user_specified_namebatch_normalization_16/gamma:;7
5
_user_specified_namebatch_normalization_16/beta:B>
<
_user_specified_name$"batch_normalization_16/moving_mean:FB
@
_user_specified_name(&batch_normalization_16/moving_variance:/+
)
_user_specified_namedense_10/kernel:- )
'
_user_specified_namedense_10/bias:<!8
6
_user_specified_namebatch_normalization_17/gamma:;"7
5
_user_specified_namebatch_normalization_17/beta:B#>
<
_user_specified_name$"batch_normalization_17/moving_mean:F$B
@
_user_specified_name(&batch_normalization_17/moving_variance:/%+
)
_user_specified_namedense_11/kernel:-&)
'
_user_specified_namedense_11/bias:)'%
#
_user_specified_name	iteration:-()
'
_user_specified_namelearning_rate:6)2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6*2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4+0
.
_user_specified_nameAdam/m/conv2d_6/bias:4,0
.
_user_specified_nameAdam/v/conv2d_6/bias:C-?
=
_user_specified_name%#Adam/m/batch_normalization_12/gamma:C.?
=
_user_specified_name%#Adam/v/batch_normalization_12/gamma:B/>
<
_user_specified_name$"Adam/m/batch_normalization_12/beta:B0>
<
_user_specified_name$"Adam/v/batch_normalization_12/beta:612
0
_user_specified_nameAdam/m/conv2d_7/kernel:622
0
_user_specified_nameAdam/v/conv2d_7/kernel:430
.
_user_specified_nameAdam/m/conv2d_7/bias:440
.
_user_specified_nameAdam/v/conv2d_7/bias:C5?
=
_user_specified_name%#Adam/m/batch_normalization_13/gamma:C6?
=
_user_specified_name%#Adam/v/batch_normalization_13/gamma:B7>
<
_user_specified_name$"Adam/m/batch_normalization_13/beta:B8>
<
_user_specified_name$"Adam/v/batch_normalization_13/beta:692
0
_user_specified_nameAdam/m/conv2d_8/kernel:6:2
0
_user_specified_nameAdam/v/conv2d_8/kernel:4;0
.
_user_specified_nameAdam/m/conv2d_8/bias:4<0
.
_user_specified_nameAdam/v/conv2d_8/bias:C=?
=
_user_specified_name%#Adam/m/batch_normalization_14/gamma:C>?
=
_user_specified_name%#Adam/v/batch_normalization_14/gamma:B?>
<
_user_specified_name$"Adam/m/batch_normalization_14/beta:B@>
<
_user_specified_name$"Adam/v/batch_normalization_14/beta:5A1
/
_user_specified_nameAdam/m/dense_8/kernel:5B1
/
_user_specified_nameAdam/v/dense_8/kernel:3C/
-
_user_specified_nameAdam/m/dense_8/bias:3D/
-
_user_specified_nameAdam/v/dense_8/bias:CE?
=
_user_specified_name%#Adam/m/batch_normalization_15/gamma:CF?
=
_user_specified_name%#Adam/v/batch_normalization_15/gamma:BG>
<
_user_specified_name$"Adam/m/batch_normalization_15/beta:BH>
<
_user_specified_name$"Adam/v/batch_normalization_15/beta:5I1
/
_user_specified_nameAdam/m/dense_9/kernel:5J1
/
_user_specified_nameAdam/v/dense_9/kernel:3K/
-
_user_specified_nameAdam/m/dense_9/bias:3L/
-
_user_specified_nameAdam/v/dense_9/bias:CM?
=
_user_specified_name%#Adam/m/batch_normalization_16/gamma:CN?
=
_user_specified_name%#Adam/v/batch_normalization_16/gamma:BO>
<
_user_specified_name$"Adam/m/batch_normalization_16/beta:BP>
<
_user_specified_name$"Adam/v/batch_normalization_16/beta:6Q2
0
_user_specified_nameAdam/m/dense_10/kernel:6R2
0
_user_specified_nameAdam/v/dense_10/kernel:4S0
.
_user_specified_nameAdam/m/dense_10/bias:4T0
.
_user_specified_nameAdam/v/dense_10/bias:CU?
=
_user_specified_name%#Adam/m/batch_normalization_17/gamma:CV?
=
_user_specified_name%#Adam/v/batch_normalization_17/gamma:BW>
<
_user_specified_name$"Adam/m/batch_normalization_17/beta:BX>
<
_user_specified_name$"Adam/v/batch_normalization_17/beta:6Y2
0
_user_specified_nameAdam/m/dense_11/kernel:6Z2
0
_user_specified_nameAdam/v/dense_11/kernel:4[0
.
_user_specified_nameAdam/m/dense_11/bias:4\0
.
_user_specified_nameAdam/v/dense_11/bias:']#
!
_user_specified_name	total_1:'^#
!
_user_specified_name	count_1:%_!

_user_specified_nametotal:%`!

_user_specified_namecount:=a9

_output_shapes
: 

_user_specified_nameConst
�

�
8__inference_batch_normalization_14_layer_call_fn_1388470

inputs
unknown:8
	unknown_0:8
	unknown_1:8
	unknown_2:8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������8*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387042�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388460:'#
!
_user_specified_name	1388462:'#
!
_user_specified_name	1388464:'#
!
_user_specified_name	1388466
�
�
*__inference_dense_10_layer_call_fn_1388825

inputs
unknown:e 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1387555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������e: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388819:'#
!
_user_specified_name	1388821
�&
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388769

inputs5
'assignmovingavg_readvariableop_resource:e7
)assignmovingavg_1_readvariableop_resource:e3
%batchnorm_mul_readvariableop_resource:e/
!batchnorm_readvariableop_resource:e
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:e*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:e�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������el
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:e*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:e*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:e*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:e*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:e�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:e*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:e~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:e�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:eP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:e~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:e*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������eh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:e*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������eb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1388444

inputs8
conv2d_readvariableop_resource:8-
biasadd_readvariableop_resource:8
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:8*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������8X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������8�
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype0�
"conv2d_8/kernel/Regularizer/L2LossL2Loss9conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0+conv2d_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������8�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
%__inference_signature_wrapper_1388150
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:8

unknown_12:8

unknown_13:8

unknown_14:8

unknown_15:8

unknown_16:8

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�e

unknown_24:e

unknown_25:e

unknown_26:e

unknown_27:e

unknown_28:e

unknown_29:e 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 


unknown_36:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1386862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:'#
!
_user_specified_name	1388072:'#
!
_user_specified_name	1388074:'#
!
_user_specified_name	1388076:'#
!
_user_specified_name	1388078:'#
!
_user_specified_name	1388080:'#
!
_user_specified_name	1388082:'#
!
_user_specified_name	1388084:'#
!
_user_specified_name	1388086:'	#
!
_user_specified_name	1388088:'
#
!
_user_specified_name	1388090:'#
!
_user_specified_name	1388092:'#
!
_user_specified_name	1388094:'#
!
_user_specified_name	1388096:'#
!
_user_specified_name	1388098:'#
!
_user_specified_name	1388100:'#
!
_user_specified_name	1388102:'#
!
_user_specified_name	1388104:'#
!
_user_specified_name	1388106:'#
!
_user_specified_name	1388108:'#
!
_user_specified_name	1388110:'#
!
_user_specified_name	1388112:'#
!
_user_specified_name	1388114:'#
!
_user_specified_name	1388116:'#
!
_user_specified_name	1388118:'#
!
_user_specified_name	1388120:'#
!
_user_specified_name	1388122:'#
!
_user_specified_name	1388124:'#
!
_user_specified_name	1388126:'#
!
_user_specified_name	1388128:'#
!
_user_specified_name	1388130:'#
!
_user_specified_name	1388132:' #
!
_user_specified_name	1388134:'!#
!
_user_specified_name	1388136:'"#
!
_user_specified_name	1388138:'##
!
_user_specified_name	1388140:'$#
!
_user_specified_name	1388142:'%#
!
_user_specified_name	1388144:'&#
!
_user_specified_name	1388146
�	
�
8__inference_batch_normalization_17_layer_call_fn_1388853

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387272o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388843:'#
!
_user_specified_name	1388845:'#
!
_user_specified_name	1388847:'#
!
_user_specified_name	1388849
�
e
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387716

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������e[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������e"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1386929

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387132

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_conv2d_7_layer_call_fn_1388306

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1387378w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388300:'#
!
_user_specified_name	1388302
�

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387405

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1387073

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388260

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
M
1__inference_max_pooling2d_8_layer_call_fn_1388511

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1387073�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_6_layer_call_fn_1388183

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387335w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388177:'#
!
_user_specified_name	1388179
�

�
8__inference_batch_normalization_12_layer_call_fn_1388211

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386880�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388201:'#
!
_user_specified_name	1388203:'#
!
_user_specified_name	1388205:'#
!
_user_specified_name	1388207
�
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388297

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_10_layer_call_and_return_conditional_losses_1387555

inputs0
matmul_readvariableop_resource:e -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:e *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:e *
dtype0�
"dense_10/kernel/Regularizer/L2LossL2Loss9dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0+dense_10/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_10/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������e: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387611
input_3*
conv2d_6_1387336:
conv2d_6_1387338:,
batch_normalization_12_1387341:,
batch_normalization_12_1387343:,
batch_normalization_12_1387345:,
batch_normalization_12_1387347:*
conv2d_7_1387379:
conv2d_7_1387381:,
batch_normalization_13_1387384:,
batch_normalization_13_1387386:,
batch_normalization_13_1387388:,
batch_normalization_13_1387390:*
conv2d_8_1387422:8
conv2d_8_1387424:8,
batch_normalization_14_1387427:8,
batch_normalization_14_1387429:8,
batch_normalization_14_1387431:8,
batch_normalization_14_1387433:8#
dense_8_1387472:
��
dense_8_1387474:	�-
batch_normalization_15_1387477:	�-
batch_normalization_15_1387479:	�-
batch_normalization_15_1387481:	�-
batch_normalization_15_1387483:	�"
dense_9_1387514:	�e
dense_9_1387516:e,
batch_normalization_16_1387519:e,
batch_normalization_16_1387521:e,
batch_normalization_16_1387523:e,
batch_normalization_16_1387525:e"
dense_10_1387556:e 
dense_10_1387558: ,
batch_normalization_17_1387561: ,
batch_normalization_17_1387563: ,
batch_normalization_17_1387565: ,
batch_normalization_17_1387567: "
dense_11_1387581: 

dense_11_1387583:

identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_7/StatefulPartitionedCall�1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp� conv2d_8/StatefulPartitionedCall�1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp� dense_10/StatefulPartitionedCall�1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp� dense_11/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp�dense_9/StatefulPartitionedCall�0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�"dropout_13/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_6_1387336conv2d_6_1387338*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387335�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_12_1387341batch_normalization_12_1387343batch_normalization_12_1387345batch_normalization_12_1387347*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386880�
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1386929�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387362�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0conv2d_7_1387379conv2d_7_1387381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1387378�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_13_1387384batch_normalization_13_1387386batch_normalization_13_1387388batch_normalization_13_1387390*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386952�
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1387001�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1387405�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0conv2d_8_1387422conv2d_8_1387424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1387421�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_14_1387427batch_normalization_14_1387429batch_normalization_14_1387431batch_normalization_14_1387433*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387024�
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1387073�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_12_layer_call_and_return_conditional_losses_1387448�
flatten_2/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1387455�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_8_1387472dense_8_1387474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1387471�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_15_1387477batch_normalization_15_1387479batch_normalization_15_1387481batch_normalization_15_1387483*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1387112�
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387497�
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_9_1387514dense_9_1387516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1387513�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_16_1387519batch_normalization_16_1387521batch_normalization_16_1387523batch_normalization_16_1387525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387192�
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387539�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_10_1387556dense_10_1387558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1387555�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_17_1387561batch_normalization_17_1387563batch_normalization_17_1387565batch_normalization_17_1387567*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1387272�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0dense_11_1387581dense_11_1387583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_1387580�
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_1387336*&
_output_shapes
:*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_7_1387379*&
_output_shapes
:*
dtype0�
"conv2d_7/kernel/Regularizer/L2LossL2Loss9conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0+conv2d_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_8_1387422*&
_output_shapes
:8*
dtype0�
"conv2d_8/kernel/Regularizer/L2LossL2Loss9conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0+conv2d_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_8_1387472* 
_output_shapes
:
��*
dtype0�
!dense_8/kernel/Regularizer/L2LossL2Loss8dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_8/kernel/Regularizer/mulMul)dense_8/kernel/Regularizer/mul/x:output:0*dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_9_1387514*
_output_shapes
:	�e*
dtype0�
!dense_9/kernel/Regularizer/L2LossL2Loss8dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0*dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_10_1387556*
_output_shapes

:e *
dtype0�
"dense_10/kernel/Regularizer/L2LossL2Loss9dense_10/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_10/kernel/Regularizer/mulMul*dense_10/kernel/Regularizer/mul/x:output:0+dense_10/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall2^conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_8/StatefulPartitionedCall2^conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_10/StatefulPartitionedCall2^dense_10/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall1^dense_8/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_9/StatefulPartitionedCall1^dense_9/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2f
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2f
1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_8/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2f
1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp1dense_10/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2d
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2d
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:'#
!
_user_specified_name	1387336:'#
!
_user_specified_name	1387338:'#
!
_user_specified_name	1387341:'#
!
_user_specified_name	1387343:'#
!
_user_specified_name	1387345:'#
!
_user_specified_name	1387347:'#
!
_user_specified_name	1387379:'#
!
_user_specified_name	1387381:'	#
!
_user_specified_name	1387384:'
#
!
_user_specified_name	1387386:'#
!
_user_specified_name	1387388:'#
!
_user_specified_name	1387390:'#
!
_user_specified_name	1387422:'#
!
_user_specified_name	1387424:'#
!
_user_specified_name	1387427:'#
!
_user_specified_name	1387429:'#
!
_user_specified_name	1387431:'#
!
_user_specified_name	1387433:'#
!
_user_specified_name	1387472:'#
!
_user_specified_name	1387474:'#
!
_user_specified_name	1387477:'#
!
_user_specified_name	1387479:'#
!
_user_specified_name	1387481:'#
!
_user_specified_name	1387483:'#
!
_user_specified_name	1387514:'#
!
_user_specified_name	1387516:'#
!
_user_specified_name	1387519:'#
!
_user_specified_name	1387521:'#
!
_user_specified_name	1387523:'#
!
_user_specified_name	1387525:'#
!
_user_specified_name	1387556:' #
!
_user_specified_name	1387558:'!#
!
_user_specified_name	1387561:'"#
!
_user_specified_name	1387563:'##
!
_user_specified_name	1387565:'$#
!
_user_specified_name	1387567:'%#
!
_user_specified_name	1387581:'&#
!
_user_specified_name	1387583
�
�
D__inference_dense_9_layer_call_and_return_conditional_losses_1388709

inputs1
matmul_readvariableop_resource:	�e-
biasadd_readvariableop_resource:e
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�e*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������er
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:e*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������eP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������e�
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�e*
dtype0�
!dense_9/kernel/Regularizer/L2LossL2Loss8dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_9/kernel/Regularizer/mulMul)dense_9/kernel/Regularizer/mul/x:output:0*dense_9/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_9/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp0dense_9/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

f
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388680

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1387362

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387192

inputs5
'assignmovingavg_readvariableop_resource:e7
)assignmovingavg_1_readvariableop_resource:e3
%batchnorm_mul_readvariableop_resource:e/
!batchnorm_readvariableop_resource:e
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:e*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:e�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������el
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:e*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:e*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:e*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:e*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ex
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:e�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:e*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:e~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:e�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:eP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:e~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:e*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������eh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:ev
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:e*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������eb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_dense_11_layer_call_fn_1388929

inputs
unknown: 

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_11_layer_call_and_return_conditional_losses_1387580o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388923:'#
!
_user_specified_name	1388925
�
�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387212

inputs/
!batchnorm_readvariableop_resource:e3
%batchnorm_mul_readvariableop_resource:e1
#batchnorm_readvariableop_1_resource:e1
#batchnorm_readvariableop_2_resource:e
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:e*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:eP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:e~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:e*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ec
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������ez
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:e*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:ez
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:e*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:er
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������eb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������e�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1387024

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������8:8:8:8:8:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
8__inference_batch_normalization_13_layer_call_fn_1388347

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386970�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388337:'#
!
_user_specified_name	1388339:'#
!
_user_specified_name	1388341:'#
!
_user_specified_name	1388343
�	
�
__inference_loss_fn_1_1388956T
:conv2d_7_kernel_regularizer_l2loss_readvariableop_resource:
identity��1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp�
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_7_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_7/kernel/Regularizer/L2LossL2Loss9conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0+conv2d_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_7/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388292

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *<|�?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��~=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_1388948T
:conv2d_6_kernel_regularizer_l2loss_readvariableop_resource:
identity��1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp�
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_6_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
H
,__inference_dropout_14_layer_call_fn_1388799

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_1387716`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������e"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������e:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs
�
e
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388543

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������8c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������8"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
��
�?
#__inference__traced_restore_1389883
file_prefix:
 assignvariableop_conv2d_6_kernel:.
 assignvariableop_1_conv2d_6_bias:=
/assignvariableop_2_batch_normalization_12_gamma:<
.assignvariableop_3_batch_normalization_12_beta:C
5assignvariableop_4_batch_normalization_12_moving_mean:G
9assignvariableop_5_batch_normalization_12_moving_variance:<
"assignvariableop_6_conv2d_7_kernel:.
 assignvariableop_7_conv2d_7_bias:=
/assignvariableop_8_batch_normalization_13_gamma:<
.assignvariableop_9_batch_normalization_13_beta:D
6assignvariableop_10_batch_normalization_13_moving_mean:H
:assignvariableop_11_batch_normalization_13_moving_variance:=
#assignvariableop_12_conv2d_8_kernel:8/
!assignvariableop_13_conv2d_8_bias:8>
0assignvariableop_14_batch_normalization_14_gamma:8=
/assignvariableop_15_batch_normalization_14_beta:8D
6assignvariableop_16_batch_normalization_14_moving_mean:8H
:assignvariableop_17_batch_normalization_14_moving_variance:86
"assignvariableop_18_dense_8_kernel:
��/
 assignvariableop_19_dense_8_bias:	�?
0assignvariableop_20_batch_normalization_15_gamma:	�>
/assignvariableop_21_batch_normalization_15_beta:	�E
6assignvariableop_22_batch_normalization_15_moving_mean:	�I
:assignvariableop_23_batch_normalization_15_moving_variance:	�5
"assignvariableop_24_dense_9_kernel:	�e.
 assignvariableop_25_dense_9_bias:e>
0assignvariableop_26_batch_normalization_16_gamma:e=
/assignvariableop_27_batch_normalization_16_beta:eD
6assignvariableop_28_batch_normalization_16_moving_mean:eH
:assignvariableop_29_batch_normalization_16_moving_variance:e5
#assignvariableop_30_dense_10_kernel:e /
!assignvariableop_31_dense_10_bias: >
0assignvariableop_32_batch_normalization_17_gamma: =
/assignvariableop_33_batch_normalization_17_beta: D
6assignvariableop_34_batch_normalization_17_moving_mean: H
:assignvariableop_35_batch_normalization_17_moving_variance: 5
#assignvariableop_36_dense_11_kernel: 
/
!assignvariableop_37_dense_11_bias:
'
assignvariableop_38_iteration:	 +
!assignvariableop_39_learning_rate: D
*assignvariableop_40_adam_m_conv2d_6_kernel:D
*assignvariableop_41_adam_v_conv2d_6_kernel:6
(assignvariableop_42_adam_m_conv2d_6_bias:6
(assignvariableop_43_adam_v_conv2d_6_bias:E
7assignvariableop_44_adam_m_batch_normalization_12_gamma:E
7assignvariableop_45_adam_v_batch_normalization_12_gamma:D
6assignvariableop_46_adam_m_batch_normalization_12_beta:D
6assignvariableop_47_adam_v_batch_normalization_12_beta:D
*assignvariableop_48_adam_m_conv2d_7_kernel:D
*assignvariableop_49_adam_v_conv2d_7_kernel:6
(assignvariableop_50_adam_m_conv2d_7_bias:6
(assignvariableop_51_adam_v_conv2d_7_bias:E
7assignvariableop_52_adam_m_batch_normalization_13_gamma:E
7assignvariableop_53_adam_v_batch_normalization_13_gamma:D
6assignvariableop_54_adam_m_batch_normalization_13_beta:D
6assignvariableop_55_adam_v_batch_normalization_13_beta:D
*assignvariableop_56_adam_m_conv2d_8_kernel:8D
*assignvariableop_57_adam_v_conv2d_8_kernel:86
(assignvariableop_58_adam_m_conv2d_8_bias:86
(assignvariableop_59_adam_v_conv2d_8_bias:8E
7assignvariableop_60_adam_m_batch_normalization_14_gamma:8E
7assignvariableop_61_adam_v_batch_normalization_14_gamma:8D
6assignvariableop_62_adam_m_batch_normalization_14_beta:8D
6assignvariableop_63_adam_v_batch_normalization_14_beta:8=
)assignvariableop_64_adam_m_dense_8_kernel:
��=
)assignvariableop_65_adam_v_dense_8_kernel:
��6
'assignvariableop_66_adam_m_dense_8_bias:	�6
'assignvariableop_67_adam_v_dense_8_bias:	�F
7assignvariableop_68_adam_m_batch_normalization_15_gamma:	�F
7assignvariableop_69_adam_v_batch_normalization_15_gamma:	�E
6assignvariableop_70_adam_m_batch_normalization_15_beta:	�E
6assignvariableop_71_adam_v_batch_normalization_15_beta:	�<
)assignvariableop_72_adam_m_dense_9_kernel:	�e<
)assignvariableop_73_adam_v_dense_9_kernel:	�e5
'assignvariableop_74_adam_m_dense_9_bias:e5
'assignvariableop_75_adam_v_dense_9_bias:eE
7assignvariableop_76_adam_m_batch_normalization_16_gamma:eE
7assignvariableop_77_adam_v_batch_normalization_16_gamma:eD
6assignvariableop_78_adam_m_batch_normalization_16_beta:eD
6assignvariableop_79_adam_v_batch_normalization_16_beta:e<
*assignvariableop_80_adam_m_dense_10_kernel:e <
*assignvariableop_81_adam_v_dense_10_kernel:e 6
(assignvariableop_82_adam_m_dense_10_bias: 6
(assignvariableop_83_adam_v_dense_10_bias: E
7assignvariableop_84_adam_m_batch_normalization_17_gamma: E
7assignvariableop_85_adam_v_batch_normalization_17_gamma: D
6assignvariableop_86_adam_m_batch_normalization_17_beta: D
6assignvariableop_87_adam_v_batch_normalization_17_beta: <
*assignvariableop_88_adam_m_dense_11_kernel: 
<
*assignvariableop_89_adam_v_dense_11_kernel: 
6
(assignvariableop_90_adam_m_dense_11_bias:
6
(assignvariableop_91_adam_v_dense_11_bias:
%
assignvariableop_92_total_1: %
assignvariableop_93_count_1: #
assignvariableop_94_total: #
assignvariableop_95_count: 
identity_97��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�)
value�(B�(aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�
value�B�aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_12_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_12_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_12_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_12_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_13_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_13_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_13_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_13_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_8_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_8_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_14_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_14_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_14_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_14_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_8_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_8_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_15_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_15_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_15_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_15_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_9_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_9_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_16_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_16_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_16_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_16_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_10_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_10_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_17_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_17_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_17_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_17_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_11_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_11_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_iterationIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp!assignvariableop_39_learning_rateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_m_conv2d_6_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_v_conv2d_6_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_m_conv2d_6_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_v_conv2d_6_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_m_batch_normalization_12_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_v_batch_normalization_12_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_m_batch_normalization_12_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_v_batch_normalization_12_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_conv2d_7_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_conv2d_7_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_conv2d_7_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_conv2d_7_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_m_batch_normalization_13_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_v_batch_normalization_13_gammaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_m_batch_normalization_13_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_v_batch_normalization_13_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_m_conv2d_8_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_v_conv2d_8_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_m_conv2d_8_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_v_conv2d_8_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_m_batch_normalization_14_gammaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_v_batch_normalization_14_gammaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_m_batch_normalization_14_betaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_v_batch_normalization_14_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_m_dense_8_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adam_v_dense_8_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adam_m_dense_8_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_v_dense_8_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_m_batch_normalization_15_gammaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_v_batch_normalization_15_gammaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_m_batch_normalization_15_betaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp6assignvariableop_71_adam_v_batch_normalization_15_betaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_m_dense_9_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_v_dense_9_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_m_dense_9_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_v_dense_9_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_m_batch_normalization_16_gammaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_v_batch_normalization_16_gammaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_m_batch_normalization_16_betaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_v_batch_normalization_16_betaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_m_dense_10_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_v_dense_10_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_m_dense_10_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_v_dense_10_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_m_batch_normalization_17_gammaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_v_batch_normalization_17_gammaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_m_batch_normalization_17_betaIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp6assignvariableop_87_adam_v_batch_normalization_17_betaIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_m_dense_11_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_v_dense_11_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_m_dense_11_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_v_dense_11_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOpassignvariableop_92_total_1Identity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOpassignvariableop_93_count_1Identity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOpassignvariableop_94_totalIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOpassignvariableop_95_countIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_96Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_97IdentityIdentity_96:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95*
_output_shapes
 "#
identity_97Identity_97:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_95:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:<8
6
_user_specified_namebatch_normalization_12/gamma:;7
5
_user_specified_namebatch_normalization_12/beta:B>
<
_user_specified_name$"batch_normalization_12/moving_mean:FB
@
_user_specified_name(&batch_normalization_12/moving_variance:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:<	8
6
_user_specified_namebatch_normalization_13/gamma:;
7
5
_user_specified_namebatch_normalization_13/beta:B>
<
_user_specified_name$"batch_normalization_13/moving_mean:FB
@
_user_specified_name(&batch_normalization_13/moving_variance:/+
)
_user_specified_nameconv2d_8/kernel:-)
'
_user_specified_nameconv2d_8/bias:<8
6
_user_specified_namebatch_normalization_14/gamma:;7
5
_user_specified_namebatch_normalization_14/beta:B>
<
_user_specified_name$"batch_normalization_14/moving_mean:FB
@
_user_specified_name(&batch_normalization_14/moving_variance:.*
(
_user_specified_namedense_8/kernel:,(
&
_user_specified_namedense_8/bias:<8
6
_user_specified_namebatch_normalization_15/gamma:;7
5
_user_specified_namebatch_normalization_15/beta:B>
<
_user_specified_name$"batch_normalization_15/moving_mean:FB
@
_user_specified_name(&batch_normalization_15/moving_variance:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_9/bias:<8
6
_user_specified_namebatch_normalization_16/gamma:;7
5
_user_specified_namebatch_normalization_16/beta:B>
<
_user_specified_name$"batch_normalization_16/moving_mean:FB
@
_user_specified_name(&batch_normalization_16/moving_variance:/+
)
_user_specified_namedense_10/kernel:- )
'
_user_specified_namedense_10/bias:<!8
6
_user_specified_namebatch_normalization_17/gamma:;"7
5
_user_specified_namebatch_normalization_17/beta:B#>
<
_user_specified_name$"batch_normalization_17/moving_mean:F$B
@
_user_specified_name(&batch_normalization_17/moving_variance:/%+
)
_user_specified_namedense_11/kernel:-&)
'
_user_specified_namedense_11/bias:)'%
#
_user_specified_name	iteration:-()
'
_user_specified_namelearning_rate:6)2
0
_user_specified_nameAdam/m/conv2d_6/kernel:6*2
0
_user_specified_nameAdam/v/conv2d_6/kernel:4+0
.
_user_specified_nameAdam/m/conv2d_6/bias:4,0
.
_user_specified_nameAdam/v/conv2d_6/bias:C-?
=
_user_specified_name%#Adam/m/batch_normalization_12/gamma:C.?
=
_user_specified_name%#Adam/v/batch_normalization_12/gamma:B/>
<
_user_specified_name$"Adam/m/batch_normalization_12/beta:B0>
<
_user_specified_name$"Adam/v/batch_normalization_12/beta:612
0
_user_specified_nameAdam/m/conv2d_7/kernel:622
0
_user_specified_nameAdam/v/conv2d_7/kernel:430
.
_user_specified_nameAdam/m/conv2d_7/bias:440
.
_user_specified_nameAdam/v/conv2d_7/bias:C5?
=
_user_specified_name%#Adam/m/batch_normalization_13/gamma:C6?
=
_user_specified_name%#Adam/v/batch_normalization_13/gamma:B7>
<
_user_specified_name$"Adam/m/batch_normalization_13/beta:B8>
<
_user_specified_name$"Adam/v/batch_normalization_13/beta:692
0
_user_specified_nameAdam/m/conv2d_8/kernel:6:2
0
_user_specified_nameAdam/v/conv2d_8/kernel:4;0
.
_user_specified_nameAdam/m/conv2d_8/bias:4<0
.
_user_specified_nameAdam/v/conv2d_8/bias:C=?
=
_user_specified_name%#Adam/m/batch_normalization_14/gamma:C>?
=
_user_specified_name%#Adam/v/batch_normalization_14/gamma:B?>
<
_user_specified_name$"Adam/m/batch_normalization_14/beta:B@>
<
_user_specified_name$"Adam/v/batch_normalization_14/beta:5A1
/
_user_specified_nameAdam/m/dense_8/kernel:5B1
/
_user_specified_nameAdam/v/dense_8/kernel:3C/
-
_user_specified_nameAdam/m/dense_8/bias:3D/
-
_user_specified_nameAdam/v/dense_8/bias:CE?
=
_user_specified_name%#Adam/m/batch_normalization_15/gamma:CF?
=
_user_specified_name%#Adam/v/batch_normalization_15/gamma:BG>
<
_user_specified_name$"Adam/m/batch_normalization_15/beta:BH>
<
_user_specified_name$"Adam/v/batch_normalization_15/beta:5I1
/
_user_specified_nameAdam/m/dense_9/kernel:5J1
/
_user_specified_nameAdam/v/dense_9/kernel:3K/
-
_user_specified_nameAdam/m/dense_9/bias:3L/
-
_user_specified_nameAdam/v/dense_9/bias:CM?
=
_user_specified_name%#Adam/m/batch_normalization_16/gamma:CN?
=
_user_specified_name%#Adam/v/batch_normalization_16/gamma:BO>
<
_user_specified_name$"Adam/m/batch_normalization_16/beta:BP>
<
_user_specified_name$"Adam/v/batch_normalization_16/beta:6Q2
0
_user_specified_nameAdam/m/dense_10/kernel:6R2
0
_user_specified_nameAdam/v/dense_10/kernel:4S0
.
_user_specified_nameAdam/m/dense_10/bias:4T0
.
_user_specified_nameAdam/v/dense_10/bias:CU?
=
_user_specified_name%#Adam/m/batch_normalization_17/gamma:CV?
=
_user_specified_name%#Adam/v/batch_normalization_17/gamma:BW>
<
_user_specified_name$"Adam/m/batch_normalization_17/beta:BX>
<
_user_specified_name$"Adam/v/batch_normalization_17/beta:6Y2
0
_user_specified_nameAdam/m/dense_11/kernel:6Z2
0
_user_specified_nameAdam/v/dense_11/kernel:4[0
.
_user_specified_nameAdam/m/dense_11/bias:4\0
.
_user_specified_nameAdam/v/dense_11/bias:']#
!
_user_specified_name	total_1:'^#
!
_user_specified_name	count_1:%_!

_user_specified_nametotal:%`!

_user_specified_namecount
�
�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388242

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1386952

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1386898

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
H
,__inference_dropout_13_layer_call_fn_1388668

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387696a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_13_layer_call_and_return_conditional_losses_1387696

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_11_layer_call_and_return_conditional_losses_1388940

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1387001

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
.__inference_sequential_2_layer_call_fn_1387843
input_3!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:8

unknown_12:8

unknown_13:8

unknown_14:8

unknown_15:8

unknown_16:8

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�e

unknown_24:e

unknown_25:e

unknown_26:e

unknown_27:e

unknown_28:e

unknown_29:e 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 


unknown_36:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*<
_read_only_resource_inputs
	
 #$%&*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387611o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3:'#
!
_user_specified_name	1387765:'#
!
_user_specified_name	1387767:'#
!
_user_specified_name	1387769:'#
!
_user_specified_name	1387771:'#
!
_user_specified_name	1387773:'#
!
_user_specified_name	1387775:'#
!
_user_specified_name	1387777:'#
!
_user_specified_name	1387779:'	#
!
_user_specified_name	1387781:'
#
!
_user_specified_name	1387783:'#
!
_user_specified_name	1387785:'#
!
_user_specified_name	1387787:'#
!
_user_specified_name	1387789:'#
!
_user_specified_name	1387791:'#
!
_user_specified_name	1387793:'#
!
_user_specified_name	1387795:'#
!
_user_specified_name	1387797:'#
!
_user_specified_name	1387799:'#
!
_user_specified_name	1387801:'#
!
_user_specified_name	1387803:'#
!
_user_specified_name	1387805:'#
!
_user_specified_name	1387807:'#
!
_user_specified_name	1387809:'#
!
_user_specified_name	1387811:'#
!
_user_specified_name	1387813:'#
!
_user_specified_name	1387815:'#
!
_user_specified_name	1387817:'#
!
_user_specified_name	1387819:'#
!
_user_specified_name	1387821:'#
!
_user_specified_name	1387823:'#
!
_user_specified_name	1387825:' #
!
_user_specified_name	1387827:'!#
!
_user_specified_name	1387829:'"#
!
_user_specified_name	1387831:'##
!
_user_specified_name	1387833:'$#
!
_user_specified_name	1387835:'%#
!
_user_specified_name	1387837:'&#
!
_user_specified_name	1387839
�
�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388488

inputs%
readvariableop_resource:8'
readvariableop_1_resource:86
(fusedbatchnormv3_readvariableop_resource:88
*fusedbatchnormv3_readvariableop_1_resource:8
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������8:8:8:8:8:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������8�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������8: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������8
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
8__inference_batch_normalization_16_layer_call_fn_1388735

inputs
unknown:e
	unknown_0:e
	unknown_1:e
	unknown_2:e
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������e*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1387212o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������e<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������e: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������e
 
_user_specified_nameinputs:'#
!
_user_specified_name	1388725:'#
!
_user_specified_name	1388727:'#
!
_user_specified_name	1388729:'#
!
_user_specified_name	1388731
�
M
1__inference_max_pooling2d_7_layer_call_fn_1388388

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1387001�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1388270

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
G
+__inference_flatten_2_layer_call_fn_1388548

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_1387455a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������8:W S
/
_output_shapes
:���������8
 
_user_specified_nameinputs
�
�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387335

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:����������
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_dense_8_layer_call_and_return_conditional_losses_1387471

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!dense_8/kernel/Regularizer/L2LossL2Loss8dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�q!>�
dense_8/kernel/Regularizer/mulMul)dense_8/kernel/Regularizer/mul/x:output:0*dense_8/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_8/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp0dense_8/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_38
serving_default_input_3:0���������  <
dense_110
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/axis
	0gamma
1beta
2moving_mean
3moving_variance"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_random_generator"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
&0
'1
02
13
24
35
G6
H7
Q8
R9
S10
T11
h12
i13
r14
s15
t16
u17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
�
&0
'1
02
13
G4
H5
Q6
R7
h8
i9
r10
s11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_sequential_2_layer_call_fn_1387843
.__inference_sequential_2_layer_call_fn_1387924�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387611
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387762�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
"__inference__wrapped_model_1386862input_3"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_6_layer_call_fn_1388183�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1388198�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
):'2conv2d_6/kernel
:2conv2d_6/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_12_layer_call_fn_1388211
8__inference_batch_normalization_12_layer_call_fn_1388224�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388242
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388260�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_12/gamma
):'2batch_normalization_12/beta
2:0 (2"batch_normalization_12/moving_mean
6:4 (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_6_layer_call_fn_1388265�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1388270�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_10_layer_call_fn_1388275
,__inference_dropout_10_layer_call_fn_1388280�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388292
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388297�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_7_layer_call_fn_1388306�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1388321�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
):'2conv2d_7/kernel
:2conv2d_7/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_13_layer_call_fn_1388334
8__inference_batch_normalization_13_layer_call_fn_1388347�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388365
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388383�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_13/gamma
):'2batch_normalization_13/beta
2:0 (2"batch_normalization_13/moving_mean
6:4 (2&batch_normalization_13/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_7_layer_call_fn_1388388�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1388393�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_11_layer_call_fn_1388398
,__inference_dropout_11_layer_call_fn_1388403�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388415
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388420�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_8_layer_call_fn_1388429�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1388444�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
):'82conv2d_8/kernel
:82conv2d_8/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
r0
s1
t2
u3"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_14_layer_call_fn_1388457
8__inference_batch_normalization_14_layer_call_fn_1388470�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388488
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388506�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(82batch_normalization_14/gamma
):'82batch_normalization_14/beta
2:08 (2"batch_normalization_14/moving_mean
6:48 (2&batch_normalization_14/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_8_layer_call_fn_1388511�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1388516�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_12_layer_call_fn_1388521
,__inference_dropout_12_layer_call_fn_1388526�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388538
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388543�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_2_layer_call_fn_1388548�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_2_layer_call_and_return_conditional_losses_1388554�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_8_layer_call_fn_1388563�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_1388578�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
": 
��2dense_8/kernel
:�2dense_8/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_15_layer_call_fn_1388591
8__inference_batch_normalization_15_layer_call_fn_1388604�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388638
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388658�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_15/gamma
*:(�2batch_normalization_15/beta
3:1� (2"batch_normalization_15/moving_mean
7:5� (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_13_layer_call_fn_1388663
,__inference_dropout_13_layer_call_fn_1388668�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388680
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388685�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_9_layer_call_fn_1388694�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_9_layer_call_and_return_conditional_losses_1388709�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
!:	�e2dense_9/kernel
:e2dense_9/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_16_layer_call_fn_1388722
8__inference_batch_normalization_16_layer_call_fn_1388735�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388769
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388789�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(e2batch_normalization_16/gamma
):'e2batch_normalization_16/beta
2:0e (2"batch_normalization_16/moving_mean
6:4e (2&batch_normalization_16/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_14_layer_call_fn_1388794
,__inference_dropout_14_layer_call_fn_1388799�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388811
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388816�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_10_layer_call_fn_1388825�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_10_layer_call_and_return_conditional_losses_1388840�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
!:e 2dense_10/kernel
: 2dense_10/bias
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_17_layer_call_fn_1388853
8__inference_batch_normalization_17_layer_call_fn_1388866�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388900
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388920�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_17/gamma
):' 2batch_normalization_17/beta
2:0  (2"batch_normalization_17/moving_mean
6:4  (2&batch_normalization_17/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_11_layer_call_fn_1388929�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_11_layer_call_and_return_conditional_losses_1388940�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
!: 
2dense_11/kernel
:
2dense_11/bias
�
�trace_02�
__inference_loss_fn_0_1388948�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_1388956�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_1388964�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_1388972�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_1388980�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_1388988�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
|
20
31
S2
T3
t4
u5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_2_layer_call_fn_1387843input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_2_layer_call_fn_1387924input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387611input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387762input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_1388150input_3"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_3
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_6_layer_call_fn_1388183inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1388198inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_12_layer_call_fn_1388211inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_12_layer_call_fn_1388224inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388242inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388260inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_6_layer_call_fn_1388265inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1388270inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_10_layer_call_fn_1388275inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_10_layer_call_fn_1388280inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388292inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388297inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_7_layer_call_fn_1388306inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1388321inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_13_layer_call_fn_1388334inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_13_layer_call_fn_1388347inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388365inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388383inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_7_layer_call_fn_1388388inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1388393inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_11_layer_call_fn_1388398inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_11_layer_call_fn_1388403inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388415inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388420inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_8_layer_call_fn_1388429inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1388444inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_14_layer_call_fn_1388457inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_14_layer_call_fn_1388470inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388488inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388506inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_8_layer_call_fn_1388511inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1388516inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_12_layer_call_fn_1388521inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_12_layer_call_fn_1388526inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388538inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388543inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_2_layer_call_fn_1388548inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_flatten_2_layer_call_and_return_conditional_losses_1388554inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_8_layer_call_fn_1388563inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_8_layer_call_and_return_conditional_losses_1388578inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_15_layer_call_fn_1388591inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_15_layer_call_fn_1388604inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388638inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388658inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_13_layer_call_fn_1388663inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_13_layer_call_fn_1388668inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388680inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388685inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_9_layer_call_fn_1388694inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_9_layer_call_and_return_conditional_losses_1388709inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_16_layer_call_fn_1388722inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_16_layer_call_fn_1388735inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388769inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388789inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_14_layer_call_fn_1388794inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_14_layer_call_fn_1388799inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388811inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388816inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_10_layer_call_fn_1388825inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_10_layer_call_and_return_conditional_losses_1388840inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_17_layer_call_fn_1388853inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_17_layer_call_fn_1388866inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388900inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388920inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_11_layer_call_fn_1388929inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_11_layer_call_and_return_conditional_losses_1388940inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
__inference_loss_fn_0_1388948"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_1388956"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_1388964"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_1388972"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_1388980"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_1388988"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
.:,2Adam/m/conv2d_6/kernel
.:,2Adam/v/conv2d_6/kernel
 :2Adam/m/conv2d_6/bias
 :2Adam/v/conv2d_6/bias
/:-2#Adam/m/batch_normalization_12/gamma
/:-2#Adam/v/batch_normalization_12/gamma
.:,2"Adam/m/batch_normalization_12/beta
.:,2"Adam/v/batch_normalization_12/beta
.:,2Adam/m/conv2d_7/kernel
.:,2Adam/v/conv2d_7/kernel
 :2Adam/m/conv2d_7/bias
 :2Adam/v/conv2d_7/bias
/:-2#Adam/m/batch_normalization_13/gamma
/:-2#Adam/v/batch_normalization_13/gamma
.:,2"Adam/m/batch_normalization_13/beta
.:,2"Adam/v/batch_normalization_13/beta
.:,82Adam/m/conv2d_8/kernel
.:,82Adam/v/conv2d_8/kernel
 :82Adam/m/conv2d_8/bias
 :82Adam/v/conv2d_8/bias
/:-82#Adam/m/batch_normalization_14/gamma
/:-82#Adam/v/batch_normalization_14/gamma
.:,82"Adam/m/batch_normalization_14/beta
.:,82"Adam/v/batch_normalization_14/beta
':%
��2Adam/m/dense_8/kernel
':%
��2Adam/v/dense_8/kernel
 :�2Adam/m/dense_8/bias
 :�2Adam/v/dense_8/bias
0:.�2#Adam/m/batch_normalization_15/gamma
0:.�2#Adam/v/batch_normalization_15/gamma
/:-�2"Adam/m/batch_normalization_15/beta
/:-�2"Adam/v/batch_normalization_15/beta
&:$	�e2Adam/m/dense_9/kernel
&:$	�e2Adam/v/dense_9/kernel
:e2Adam/m/dense_9/bias
:e2Adam/v/dense_9/bias
/:-e2#Adam/m/batch_normalization_16/gamma
/:-e2#Adam/v/batch_normalization_16/gamma
.:,e2"Adam/m/batch_normalization_16/beta
.:,e2"Adam/v/batch_normalization_16/beta
&:$e 2Adam/m/dense_10/kernel
&:$e 2Adam/v/dense_10/kernel
 : 2Adam/m/dense_10/bias
 : 2Adam/v/dense_10/bias
/:- 2#Adam/m/batch_normalization_17/gamma
/:- 2#Adam/v/batch_normalization_17/gamma
.:, 2"Adam/m/batch_normalization_17/beta
.:, 2"Adam/v/batch_normalization_17/beta
&:$ 
2Adam/m/dense_11/kernel
&:$ 
2Adam/v/dense_11/kernel
 :
2Adam/m/dense_11/bias
 :
2Adam/v/dense_11/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_1386862�:&'0123GHQRSThirstu��������������������8�5
.�+
)�&
input_3���������  
� "3�0
.
dense_11"�
dense_11���������
�
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388242�0123Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_1388260�0123Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
8__inference_batch_normalization_12_layer_call_fn_1388211�0123Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
8__inference_batch_normalization_12_layer_call_fn_1388224�0123Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388365�QRSTQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_1388383�QRSTQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
8__inference_batch_normalization_13_layer_call_fn_1388334�QRSTQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
8__inference_batch_normalization_13_layer_call_fn_1388347�QRSTQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388488�rstuQ�N
G�D
:�7
inputs+���������������������������8
p

 
� "F�C
<�9
tensor_0+���������������������������8
� �
S__inference_batch_normalization_14_layer_call_and_return_conditional_losses_1388506�rstuQ�N
G�D
:�7
inputs+���������������������������8
p 

 
� "F�C
<�9
tensor_0+���������������������������8
� �
8__inference_batch_normalization_14_layer_call_fn_1388457�rstuQ�N
G�D
:�7
inputs+���������������������������8
p

 
� ";�8
unknown+���������������������������8�
8__inference_batch_normalization_14_layer_call_fn_1388470�rstuQ�N
G�D
:�7
inputs+���������������������������8
p 

 
� ";�8
unknown+���������������������������8�
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388638s����8�5
.�+
!�
inputs����������
p

 
� "-�*
#� 
tensor_0����������
� �
S__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1388658s����8�5
.�+
!�
inputs����������
p 

 
� "-�*
#� 
tensor_0����������
� �
8__inference_batch_normalization_15_layer_call_fn_1388591h����8�5
.�+
!�
inputs����������
p

 
� ""�
unknown�����������
8__inference_batch_normalization_15_layer_call_fn_1388604h����8�5
.�+
!�
inputs����������
p 

 
� ""�
unknown�����������
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388769q����7�4
-�*
 �
inputs���������e
p

 
� ",�)
"�
tensor_0���������e
� �
S__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1388789q����7�4
-�*
 �
inputs���������e
p 

 
� ",�)
"�
tensor_0���������e
� �
8__inference_batch_normalization_16_layer_call_fn_1388722f����7�4
-�*
 �
inputs���������e
p

 
� "!�
unknown���������e�
8__inference_batch_normalization_16_layer_call_fn_1388735f����7�4
-�*
 �
inputs���������e
p 

 
� "!�
unknown���������e�
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388900q����7�4
-�*
 �
inputs��������� 
p

 
� ",�)
"�
tensor_0��������� 
� �
S__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1388920q����7�4
-�*
 �
inputs��������� 
p 

 
� ",�)
"�
tensor_0��������� 
� �
8__inference_batch_normalization_17_layer_call_fn_1388853f����7�4
-�*
 �
inputs��������� 
p

 
� "!�
unknown��������� �
8__inference_batch_normalization_17_layer_call_fn_1388866f����7�4
-�*
 �
inputs��������� 
p 

 
� "!�
unknown��������� �
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1388198s&'7�4
-�*
(�%
inputs���������  
� "4�1
*�'
tensor_0���������
� �
*__inference_conv2d_6_layer_call_fn_1388183h&'7�4
-�*
(�%
inputs���������  
� ")�&
unknown����������
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1388321sGH7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
*__inference_conv2d_7_layer_call_fn_1388306hGH7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1388444shi7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������8
� �
*__inference_conv2d_8_layer_call_fn_1388429hhi7�4
-�*
(�%
inputs���������
� ")�&
unknown���������8�
E__inference_dense_10_layer_call_and_return_conditional_losses_1388840e��/�,
%�"
 �
inputs���������e
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_10_layer_call_fn_1388825Z��/�,
%�"
 �
inputs���������e
� "!�
unknown��������� �
E__inference_dense_11_layer_call_and_return_conditional_losses_1388940e��/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������

� �
*__inference_dense_11_layer_call_fn_1388929Z��/�,
%�"
 �
inputs��������� 
� "!�
unknown���������
�
D__inference_dense_8_layer_call_and_return_conditional_losses_1388578g��0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_8_layer_call_fn_1388563\��0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_9_layer_call_and_return_conditional_losses_1388709f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������e
� �
)__inference_dense_9_layer_call_fn_1388694[��0�-
&�#
!�
inputs����������
� "!�
unknown���������e�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388292s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
G__inference_dropout_10_layer_call_and_return_conditional_losses_1388297s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
,__inference_dropout_10_layer_call_fn_1388275h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
,__inference_dropout_10_layer_call_fn_1388280h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388415s;�8
1�.
(�%
inputs���������
p
� "4�1
*�'
tensor_0���������
� �
G__inference_dropout_11_layer_call_and_return_conditional_losses_1388420s;�8
1�.
(�%
inputs���������
p 
� "4�1
*�'
tensor_0���������
� �
,__inference_dropout_11_layer_call_fn_1388398h;�8
1�.
(�%
inputs���������
p
� ")�&
unknown����������
,__inference_dropout_11_layer_call_fn_1388403h;�8
1�.
(�%
inputs���������
p 
� ")�&
unknown����������
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388538s;�8
1�.
(�%
inputs���������8
p
� "4�1
*�'
tensor_0���������8
� �
G__inference_dropout_12_layer_call_and_return_conditional_losses_1388543s;�8
1�.
(�%
inputs���������8
p 
� "4�1
*�'
tensor_0���������8
� �
,__inference_dropout_12_layer_call_fn_1388521h;�8
1�.
(�%
inputs���������8
p
� ")�&
unknown���������8�
,__inference_dropout_12_layer_call_fn_1388526h;�8
1�.
(�%
inputs���������8
p 
� ")�&
unknown���������8�
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388680e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_13_layer_call_and_return_conditional_losses_1388685e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_13_layer_call_fn_1388663Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_13_layer_call_fn_1388668Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388811c3�0
)�&
 �
inputs���������e
p
� ",�)
"�
tensor_0���������e
� �
G__inference_dropout_14_layer_call_and_return_conditional_losses_1388816c3�0
)�&
 �
inputs���������e
p 
� ",�)
"�
tensor_0���������e
� �
,__inference_dropout_14_layer_call_fn_1388794X3�0
)�&
 �
inputs���������e
p
� "!�
unknown���������e�
,__inference_dropout_14_layer_call_fn_1388799X3�0
)�&
 �
inputs���������e
p 
� "!�
unknown���������e�
F__inference_flatten_2_layer_call_and_return_conditional_losses_1388554h7�4
-�*
(�%
inputs���������8
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_2_layer_call_fn_1388548]7�4
-�*
(�%
inputs���������8
� ""�
unknown����������E
__inference_loss_fn_0_1388948$&�

� 
� "�
unknown E
__inference_loss_fn_1_1388956$G�

� 
� "�
unknown E
__inference_loss_fn_2_1388964$h�

� 
� "�
unknown F
__inference_loss_fn_3_1388972%��

� 
� "�
unknown F
__inference_loss_fn_4_1388980%��

� 
� "�
unknown F
__inference_loss_fn_5_1388988%��

� 
� "�
unknown �
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1388270�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_6_layer_call_fn_1388265�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1388393�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_7_layer_call_fn_1388388�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_1388516�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_8_layer_call_fn_1388511�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387611�:&'0123GHQRSThirstu��������������������@�=
6�3
)�&
input_3���������  
p

 
� ",�)
"�
tensor_0���������

� �
I__inference_sequential_2_layer_call_and_return_conditional_losses_1387762�:&'0123GHQRSThirstu��������������������@�=
6�3
)�&
input_3���������  
p 

 
� ",�)
"�
tensor_0���������

� �
.__inference_sequential_2_layer_call_fn_1387843�:&'0123GHQRSThirstu��������������������@�=
6�3
)�&
input_3���������  
p

 
� "!�
unknown���������
�
.__inference_sequential_2_layer_call_fn_1387924�:&'0123GHQRSThirstu��������������������@�=
6�3
)�&
input_3���������  
p 

 
� "!�
unknown���������
�
%__inference_signature_wrapper_1388150�:&'0123GHQRSThirstu��������������������C�@
� 
9�6
4
input_3)�&
input_3���������  "3�0
.
dense_11"�
dense_11���������
