import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(FusionLayer, self).__init__()
        
        # 1. توحيد الأبعاد (Projections)
        # هنكبر الـ Body والـ Face عشان يبقوا 512 زي الـ Voice والـ Text
        self.body_proj = nn.Linear(256, 512)
        self.face_proj = nn.Linear(384, 512)
        # الـ Voice والـ Text أصلاً 512 فمش محتاجين Projection
        
        # 2. شبكة الانتباه (Attention Network)
        # دي اللي هتحسب سكور لكل حاسة بناءً على أهميتها في اللحظة دي
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 3. الـ MLP النهائي بعد الدمج الذكي
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), # بيسرع التدريب جداً
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, body_vec, face_vec, voice_vec, text_vec):
        # 1. توحيد قوة الحواس كلها لـ 512
        b_proj = self.body_proj(body_vec)
        f_proj = self.face_proj(face_vec)
        v_proj = voice_vec
        t_proj = text_vec
        
        # 2. نرصهم فوق بعض في 3D Tensor عشان ندخلهم على الـ Attention
        # الشكل هيبقى: (batch_size, 4_modalities, 512_features)
        stacked_features = torch.stack([b_proj, f_proj, v_proj, t_proj], dim=1)
        
        # 3. حساب الأوزان (Attention Weights)
        # هيطلع سكور لكل حاسة
        att_scores = self.attention(stacked_features) # Shape: (batch_size, 4, 1)
        
        # السوفت ماكس بيوزع 100% على الـ 4 حواس بناءً على الأهمية
        att_weights = F.softmax(att_scores, dim=1) 
        
        # 4. الدمج الذكي (Weighted Sum)
        # بنضرب كل حاسة في وزنها ونجمعهم مع بعض
        fused_input = torch.sum(stacked_features * att_weights, dim=1) # Shape: (batch_size, 512)
        
        # 5. ندخلهم على الـ MLP الأخير عشان نطلع الناتج النهائي
        fused_output = self.fusion(fused_input)
        
        return fused_output # 512-dim fused_vector
    #(59.22%)