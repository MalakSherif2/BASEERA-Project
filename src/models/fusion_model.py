import torch
import torch.nn as nn

class SpatioTemporalFusionModel(nn.Module):
    def __init__(self, visual_dim=768, pose_dim=34, fusion_dim=512, num_classes=3, max_seq_len=30, nhead=8, num_layers=2):
        super(SpatioTemporalFusionModel, self).__init__()
        
        # 1. Projections لتوافق الأبعاد قبل الدمج
        self.visual_proj = nn.Linear(visual_dim, fusion_dim // 2) # e.g., 384
        self.pose_proj = nn.Linear(pose_dim, fusion_dim // 2)     # e.g., 384
        
        # الناتج الإجمالي للدمج هيكون fusion_dim (مثلاً 768)
        self.fusion_dim = fusion_dim
        
        # Positional Encoding للتسلسل الزمني المشترك
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, fusion_dim))
        
        # Transformer Encoder لمعالجة الـ Fused Features عبر الزمن
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head النهائي
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, visual_feats, pose_feats):
        """
        visual_feats shape: (Batch, Seq_Len, 768)
        pose_feats shape:   (Batch, Seq_Len, 34)
        """
        batch_size, seq_len, _ = visual_feats.shape
        
        # إسقاط الميزات (Projection)
        v_proj = self.visual_proj(visual_feats) # (Batch, Seq_Len, 384)
        p_proj = self.pose_proj(pose_feats)     # (Batch, Seq_Len, 384)
        
        # دمج الـ Visual مع الـ Pose (Concatenation)
        fused = torch.cat([v_proj, p_proj], dim=-1) # (Batch, Seq_Len, 768)
        
        # إضافة الـ Positional Encoding
        fused = fused + self.positional_encoding[:, :seq_len, :]
        
        # مرور البيانات على الـ Temporal Transformer
        encoded = self.transformer_encoder(fused)
        
        # Pooling وتصنيف السلوك
        pooled = torch.mean(encoded, dim=1)
        out = self.classifier(pooled)
        return out

# اختبار سريع للتأكد من نجاح الـ Fusion
if __name__ == "__main__":
    model = SpatioTemporalFusionModel()
    dummy_visual = torch.randn(1, 30, 768)
    dummy_pose = torch.randn(1, 30, 34)
    output = model(dummy_visual, dummy_pose)
    print(f"✅ Spatio-Temporal Fusion Successful! Output Shape: {output.shape}")