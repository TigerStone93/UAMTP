import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

# ========================================================================================== #

class SpectralNormalizedConvolutionalStart(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2):
        super(SpectralNormalizedConvolutionalStart, self).__init__()
        self.convolutional = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False))
        self.batch_normalization = nn.BatchNorm2d(out_channels)

        self.leaky_relu = F.leaky_relu

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Halve size of feature map for efficiency, BUT Double receptive field to make one node(neuron) see broader space.
        # self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0) # Almost maintain size of feature map to maintain spatial info, BUT Maintain receptive field to make one node(neuron) see narrower space.
        
    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.batch_normalization(self.convolutional(x))) # relu
        out = self.max_pooling(out)
        return out
        
# ========================================================================================== #

class SpectralNormalizedConvolutionalBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1): # kernel_size x kernel_size = size_of_filter x size_of_filter, in_channels = depth_of_filter, out_channels = number_of_filters
        super(SpectralNormalizedConvolutionalBlock, self).__init__()
        self.convolutional_1 = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False))
        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)
        self.convolutional_2 = utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False))
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                                          nn.BatchNorm2d(out_channels))

        self.leaky_relu = F.leaky_relu

    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.batch_normalization_1(self.convolutional_1(x))) # relu
        out = self.batch_normalization_2(self.convolutional_2(out))

        out += self.shortcut(x)

        out = self.leaky_relu(out) # relu
        return out

# ========================================================================================== #

class CrossAttention(nn.Module):
    def __init__(self, query_dim=256, key_value_dim=30, hidden_dim=256):
        super(CrossAttention, self).__init__()
        self.query_linear = utils.spectral_norm(nn.Linear(query_dim, hidden_dim))
        self.key_linear = utils.spectral_norm(nn.Linear(key_value_dim, hidden_dim))
        self.value_linear = utils.spectral_norm(nn.Linear(key_value_dim, hidden_dim))

        self.output_linear = utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))

        self.shortcut = utils.spectral_norm(nn.Linear(query_dim, hidden_dim)) # ADDED

        # Apply spectral normalization to linear layers of attention mechanism to make attention weights is distributionally spread out well,
        # which prevents entropy collapse due to low attention entropy.
        # Stabilizing Transformer Training by Preventing Attention Entropy Collapse (ICML 2023)

        # Activation function is not used in cross attention.
        
    # ============================================================ #

    def forward(self, query, key, value):
        # Linear transformation
        Q = self.query_linear(query) # (batch_size, query_dim) -> (batch_size, hidden_dim)
        K = self.key_linear(key) # (batch_size, key_value_dim) -> (batch_size, hidden_dim)
        V = self.value_linear(value) # (batch_size, key_value_dim) -> (batch_size, hidden_dim)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5) # Dot Product # (batch_size, batch_size)
        #print("[Debug] attention_scores :", attention_scores.shape)
        attention_weights = F.softmax(attention_scores, dim=-1) # Normalize to 0 ~ 1 range # (batch_size, batch_size)
        #print("[Debug] attention_weights :", attention_weights.shape)

        attention_output = torch.matmul(attention_weights, V) # (batch_size, batch_size) * (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        #print("[Debug] attention_output :", attention_output.shape)

        out = self.output_linear(attention_output) # (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        #print("[Debug] out :", out.shape)

        residual = self.shortcut(query) # ADDED # (batch_size, query_dim) -> (batch_size, hidden_dim)
        out += residual # ADDED # (batch_size, hidden_dim)
        #print("[Debug] final out :", out.shape)

        return out

# ========================================================================================== #

class SpectralNormalizedFullyConnectedBlock(nn.Module):
    expansion = 1
    def __init__(self, in_features, out_features):
        super(SpectralNormalizedFullyConnectedBlock, self).__init__()
        self.fully_connected_1 = utils.spectral_norm(nn.Linear(in_features, out_features))
        self.fully_connected_2 = utils.spectral_norm(nn.Linear(out_features, out_features))
        
        self.shortcut = utils.spectral_norm(nn.Linear(in_features, out_features)) # skip_connection
        
        self.leaky_relu = F.leaky_relu
        
    # ============================================================ #
    
    def forward(self, x):
        out = self.leaky_relu(self.fully_connected_1(x)) # relu
        out = self.fully_connected_2(out)

        residual = x
        if x.shape != out.shape:
            residual = self.shortcut(x)
        out += residual
        
        out = self.leaky_relu(out) # relu
        return out

# ========================================================================================== #

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # Convolutional Layers
        # 320x320

        self.convolutional_block_0 = SpectralNormalizedConvolutionalStart(3, 32, 7, 2) # 7x7 conv, avg pool
        # Feature map: 32x160x160 -> 32x80x80
        # Receptive field: 7 -> 11

        self.convolutional_block_1 = SpectralNormalizedConvolutionalBlock(32, 32, 3, 1) # 3x3 conv, 3x3 conv
        self.convolutional_block_2 = SpectralNormalizedConvolutionalBlock(32, 32, 3, 1)
        # Receptive field: 19 -> 27 -> 35 -> 43
        
        self.convolutional_block_3 = SpectralNormalizedConvolutionalBlock(32, 64, 3, 2)
        self.convolutional_block_4 = SpectralNormalizedConvolutionalBlock(64, 64, 3, 1)
        # Feature map: 64x40x40
        # Receptive field: 51 -> 67 -> 83 -> 99
        
        self.convolutional_block_5 = SpectralNormalizedConvolutionalBlock(64, 128, 3, 2)
        self.convolutional_block_6 = SpectralNormalizedConvolutionalBlock(128, 128, 3, 1)
        # Feature map: 128x20x20
        # Receptive field: 115 -> 147 -> 179 -> 211

        self.convolutional_block_7 = SpectralNormalizedConvolutionalBlock(128, 256, 3, 2)
        self.convolutional_block_8 = SpectralNormalizedConvolutionalBlock(256, 256, 3, 1)
        # Feature map: 256x10x10
        # Receptive field: 243 -> 307 -> 371 -> 435

        # Receptive Field Size: https://distill.pub/2019/computing-receptive-fields/
        # current_layer_RF_size = previous_layer_RF_size + (current_layer_kernel_size - 1) * previous_layer_total_stride
        # layer_RF_size starts with 1, AND total_stride is multiplied layer by layer
        # Receptive field should cover all area of input image.
        
        """
        # If want to apply GAP 1x1, then increase feature channels to 256
        # or apply GAP 5x5, then flatten???
        """
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # 256x1x1
        # Global average pooling prevents overfitting by reducing the number of parameters.
        
        # ============================== #

        # Cross Attention
        # Q: CNN_Feature, K&V: State
        self.cross_attention = CrossAttention(query_dim=256, key_value_dim=30, hidden_dim=256)

        # ============================== #

        # Fully Connected Layers
        """
        # If necessary, separate the fully_connected_block_1 for velocity and yaw,
        # or remove the fully_connected_block_1 itself.
        """
        self.fully_connected_block_1 = SpectralNormalizedFullyConnectedBlock(256, 256)

        self.fully_connected_block_2_velocity = SpectralNormalizedFullyConnectedBlock(256, 256)
        self.fully_connected_block_2_yaw = SpectralNormalizedFullyConnectedBlock(256, 256)

        self.fully_connected_block_3_velocity_1 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_velocity_2 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_velocity_3 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_velocity_4 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_yaw_1 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_yaw_2 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_yaw_3 = SpectralNormalizedFullyConnectedBlock(256, 128)
        self.fully_connected_block_3_yaw_4 = SpectralNormalizedFullyConnectedBlock(256, 128)

        self.fully_connected_block_4_velocity_1 = SpectralNormalizedFullyConnectedBlock(128, 25)
        self.fully_connected_block_4_velocity_2 = SpectralNormalizedFullyConnectedBlock(128, 25)
        self.fully_connected_block_4_velocity_3 = SpectralNormalizedFullyConnectedBlock(128, 25)
        self.fully_connected_block_4_velocity_4 = SpectralNormalizedFullyConnectedBlock(128, 25)
        self.fully_connected_block_4_yaw_1 = SpectralNormalizedFullyConnectedBlock(128, 71)
        self.fully_connected_block_4_yaw_2 = SpectralNormalizedFullyConnectedBlock(128, 71)
        self.fully_connected_block_4_yaw_3 = SpectralNormalizedFullyConnectedBlock(128, 71)
        self.fully_connected_block_4_yaw_4 = SpectralNormalizedFullyConnectedBlock(128, 71)
        
        self.feature = None #
        
    # ============================================================ #
    
    def forward(self, map_input_tensor, record_input_tensor):
        # Convolutional Layers
        out = self.convolutional_block_0(map_input_tensor)

        out = self.convolutional_block_1(out)
        out = self.convolutional_block_2(out)

        out = self.convolutional_block_3(out)
        out = self.convolutional_block_4(out)
        
        out = self.convolutional_block_5(out)
        out = self.convolutional_block_6(out)

        out = self.convolutional_block_7(out)
        out = self.convolutional_block_8(out)

        out = self.global_average_pooling(out)
        cnn_feature_tensor = out.reshape(out.size(0), -1) # (batch_size, 256, 1, 1) -> (batch_size, 256)

        # ============================== #

        # Cross Attention
        # Q: CNN_Feature, K&V: State
        fused_input = self.cross_attention(cnn_feature_tensor, record_input_tensor, record_input_tensor)
        
        out = self.fully_connected_block_1(fused_input)

        out_acc = self.fully_connected_block_2_velocity(out)
        out_ang = self.fully_connected_block_2_yaw(out)

        out_acc_1 = self.fully_connected_block_3_velocity_1(out_acc)
        out_acc_2 = self.fully_connected_block_3_velocity_2(out_acc)
        out_acc_3 = self.fully_connected_block_3_velocity_3(out_acc)
        out_acc_4 = self.fully_connected_block_3_velocity_4(out_acc)
        out_ang_1 = self.fully_connected_block_3_yaw_1(out_ang)
        out_ang_2 = self.fully_connected_block_3_yaw_2(out_ang)
        out_ang_3 = self.fully_connected_block_3_yaw_3(out_ang)
        out_ang_4 = self.fully_connected_block_3_yaw_4(out_ang)
        self.feature_v_1 = out_acc_1.clone().detach() # For get_embeddings()
        self.feature_v_2 = out_acc_2.clone().detach() # For get_embeddings()
        self.feature_v_3 = out_acc_3.clone().detach() # For get_embeddings()
        self.feature_v_4 = out_acc_4.clone().detach() # For get_embeddings()
        self.feature_y_1 = out_ang_1.clone().detach() # For get_embeddings()
        self.feature_y_2 = out_ang_2.clone().detach() # For get_embeddings()
        self.feature_y_3 = out_ang_3.clone().detach() # For get_embeddings()
        self.feature_y_4 = out_ang_4.clone().detach() # For get_embeddings()

        acc_1 = self.fully_connected_block_4_velocity_1(out_acc_1)
        acc_2 = self.fully_connected_block_4_velocity_2(out_acc_2)
        acc_3 = self.fully_connected_block_4_velocity_3(out_acc_3)
        acc_4 = self.fully_connected_block_4_velocity_4(out_acc_4)
        ang_1 = self.fully_connected_block_4_yaw_1(out_ang_1)
        ang_2 = self.fully_connected_block_4_yaw_2(out_ang_2)
        ang_3 = self.fully_connected_block_4_yaw_3(out_ang_3)
        ang_4 = self.fully_connected_block_4_yaw_4(out_ang_4)
        
        return acc_1, acc_2, acc_3, acc_4, ang_1, ang_2, ang_3, ang_4

# ========================================================================================== #

def resnet18(**kwargs):
    model = ResNet(**kwargs)
    return model