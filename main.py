import torch
print(torch.__version__)

# Tensors represent data in a numerical way

# Scalar - a single number, or a zero dimension tensor
scalar = torch.tensor(7)
print(scalar)

# returns the dimensions of the tensor
print(scalar.ndim)

# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())

# Vector - a single dimension tensor that can contain many numbers -> can have different values for different features
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)

# Check shape of vector
print(vector.shape)

# Matrix - as flexible as vectors, with an extra dimension
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)

# gets the output torch.Size([2, 2]) because MATRIX is 2 elements deep and 2 elements wide
print(MATRIX.shape)

# Tensor - can represent almost anything, this specific tensor has 3 dimensions
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)

# Check number of dimensions for TENSOR
print(TENSOR.ndim)

# Check shape of TENSOR - the output goes from higher dimensions to lower
print(TENSOR.shape)

"""
Scalars are often represented by a lowercase a,
Vectors as a lowercase y,
matrices as an uppercase Q,
and tensors as an uppercase X,
this is probably why X for features and y for target is the convention
"""

# RANDOM TENSORS - ml models stars out with random tensors
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

# size can be anything in torch.rand - Ex. common image shape:
# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

# Create a tensor of all zeros - useful for masking (like masking some of the values with zeros so model doesnt learn)
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

"""
there are many different tensor data types, the higher precision ones take longer to compute but can be more accurate
some of the data types, Ex. torch.cuda, are used for GPU instead of CPU
"""

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor.dtype)

# Create a random tensor with 3 rows 4 cols
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")  # will default to CPU

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

# Multiply it by 10
print(tensor * 10)

# Subtract and reassign
tensor = tensor - 10
print(tensor)

# Add and reassign
tensor = tensor + 10
print(tensor)

# Can also use torch functions
print(torch.multiply(tensor, 10))

# Element-wise matrix multiplication
print(tensor * tensor)

# Matrix multiplication
print(torch.matmul(tensor, tensor))

# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)

# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

# View tensor_A and tensor_B.T (B transposed)
print(tensor_A)
print(tensor_B.T)

# only works when A or B are transposed because of matrix multiplication and shape
print(torch.matmul(tensor_A, tensor_B.T))

# torch.mm is a shortcut for matmul
torch.mm(tensor_A, tensor_B.T)

# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)

# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2,  # in_features = matches inner dimension of input
                         out_features=6)  # out_features = describes outer value
x = tensor_A
output = linear(x)
print(f"\nInput shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")







