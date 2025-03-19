class AdaptiveMultiScaleFeatureFusionNet(nn.Module):
    """
    自适应多尺度特征融合网络
    """
    def __init__(self, in_channels=1, num_classes=2, initial_channels=32, depth=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.initial_channels = initial_channels
        self.depth = depth
        
        # 创建编码器
        self.encoder = Encoder(
            in_channels=in_channels,
            initial_channels=initial_channels,
            depth=depth
        )
        
        # 创建解码器
        self.decoder = Decoder(
            num_classes=num_classes,
            initial_channels=initial_channels,
            depth=depth
        )
        
        # 创建特征融合模块
        self.feature_fusion = AdaptiveFeatureFusion(
            initial_channels=initial_channels,
            depth=depth
        )
        
        # 创建注意力模块
        self.attention = AdaptiveAttention(
            initial_channels=initial_channels,
            depth=depth
        )
        
        # 创建损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 编码器前向传播
        encoder_features = self.encoder(x)
        
        # 特征融合
        fused_features = self.feature_fusion(encoder_features)
        
        # 注意力机制
        attention_weights = self.attention(fused_features)
        fused_features = [f * w for f, w in zip(fused_features, attention_weights)]
        
        # 解码器前向传播
        output = self.decoder(fused_features)
        
        return output 