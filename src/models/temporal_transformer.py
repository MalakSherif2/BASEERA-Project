import torch
import torch.nn as nn

class TemporalTransformerClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3, max_seq_len=30, nhead=8, num_layers=2):
        super(TemporalTransformerClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        
        # Positional Encoding عشان الموديل يفهم ترتيب الفريمات ورا بعضها
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, feature_dim))
        
        # Transformer Encoder Layer & Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head (لتصنيف السلوك: مثلاً fighting, robbery, stealing)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (Batch_size, Seq_Len, Feature_Dim) -> مثلاً (1, 30, 768)
        """
        batch_size, seq_len, _ = x.shape
        
        # إضافة الـ Positional Encoding للتسلسل الزمني
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # مرور البيانات على الـ Transformer Encoder (Self-Attention)
        encoded_features = self.transformer_encoder(x)
        
        # عمل Pooling (أخذ متوسط الـ features عبر الزمن لتجميع ملخص الحركة)
        pooled_features = torch.mean(encoded_features, dim=1)
        
        # التنبؤ بالسلوك النهائي
        out = self.classifier(pooled_features)
        return out

# اختبار سريع للتأكد من الأبعاد
if __name__ == "__main__":
    model = TemporalTransformerClassifier()
    # تجربة تسلسل وهمي لـ 30 فريم بـ ViT embeddings (768)
    dummy_sequence = torch.randn(1, 30, 768)
    output = model(dummy_sequence)
    print(f"✅ Temporal Transformer Successful! Output Shape: {output.shape}")