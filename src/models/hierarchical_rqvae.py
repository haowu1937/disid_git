import torch
import torch.nn as nn
from .rqvae import RQVAE


class HierarchicalRQVAE(RQVAE):
    def __init__(self, input_dim, hidden_dim, codebook_size, num_layers, num_shared_layers, num_domains,
                 domain_embed_init='random'):
        """
        num_shared_layers: 前N层为共享层
        num_domains: 域个数
        domain_embed_init: 'random' 或 'mean'（使用域内物品embedding均值初始化）
        """
        super().__init__(input_dim, hidden_dim, codebook_size, num_layers)
        self.num_shared_layers = num_shared_layers
        self.num_private_layers = num_layers - num_shared_layers
        self.num_domains = num_domains

        # 域嵌入: [num_domains, hidden_dim]
        if domain_embed_init == 'random':
            self.domain_embeddings = nn.Embedding(num_domains, hidden_dim)
        else:
            # 将在外部通过data统计初始化
            self.domain_embeddings = nn.Embedding(num_domains, hidden_dim)
            # 暂时不初始化，留到外部调用init_domain_embeddings方法

        # 私有层的码本（从父类继承的codebooks中，后num_private_layers层作为私有层）
        # 注意：父类codebooks已经包含了所有层，我们只需在forward中区分共享和私有处理

    def init_domain_embeddings_with_mean(self, dataloader, device):
        """用域内物品encoder输出的均值初始化域嵌入"""
        self.eval()
        all_hidden = []
        domain_ids = []
        with torch.no_grad():
            for batch in dataloader:
                embed = batch['embed'].to(device)
                domain = batch['domain_id']
                h = self.encoder(embed)  # [B, hidden_dim]
                all_hidden.append(h)
                domain_ids.append(domain)
        all_hidden = torch.cat(all_hidden, dim=0)
        domain_ids = torch.cat(domain_ids, dim=0)

        for d in range(self.num_domains):
            mask = (domain_ids == d)
            if mask.sum() > 0:
                mean_h = all_hidden[mask].mean(dim=0)
                self.domain_embeddings.weight.data[d] = mean_h
            else:
                # 如果域内无数据，随机初始化
                pass
        self.train()

    def forward(self, x, domain_ids):
        """
        x: [B, input_dim]
        domain_ids: [B] 长整型，每个样本所属域
        """
        x_orig = x
        h = self.encoder(x)  # [B, hidden_dim]

        quantized = torch.zeros_like(h)
        residual = h
        indices = []
        codebook_loss = 0.0

        # 共享层
        for layer in range(self.num_shared_layers):
            codebook = self.codebooks[layer]
            dist = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1)  # [B, K]
            idx = dist.argmin(dim=1)
            indices.append(idx)
            quantized_layer = codebook(idx)
            codebook_loss += F.mse_loss(quantized_layer.detach(), residual) + \
                             self.commitment_weight * F.mse_loss(quantized_layer, residual.detach())
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer.detach()

        # 私有层：在残差上加域嵌入
        for layer in range(self.num_shared_layers, self.num_layers):
            # 获取对应域的域嵌入
            domain_embed = self.domain_embeddings(domain_ids)  # [B, hidden_dim]
            residual_with_domain = residual + domain_embed  # 残差加域信息

            codebook = self.codebooks[layer]
            dist = torch.cdist(residual_with_domain.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1)
            idx = dist.argmin(dim=1)
            indices.append(idx)
            quantized_layer = codebook(idx)
            # commitment loss: 注意要使用原始残差（不加域）来计算？论文通常对量化后的向量与原始残差加域做commitment，我们保持与共享层一致：量化向量应接近 residual_with_domain
            codebook_loss += F.mse_loss(quantized_layer.detach(), residual_with_domain) + \
                             self.commitment_weight * F.mse_loss(quantized_layer, residual_with_domain.detach())
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer.detach()  # 下一层残差不加域

        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x_orig)
        total_loss = recon_loss + codebook_loss

        losses = {
            'recon': recon_loss.item(),
            'codebook': codebook_loss.item(),
            'total': total_loss.item()
        }

        return quantized, torch.stack(indices, dim=1), losses