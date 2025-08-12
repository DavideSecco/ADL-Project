import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class DenseCL(nn.Module):
    '''DenseCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,  # resnet
                 neck=None, 
                 head=None, 
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 **kwargs):
        super(DenseCL, self).__init__()

        # query encoder [backbone, neck]
        self.encoder_q = nn.Sequential(builder.build_backbone(backbone), builder.build_neck(neck))
        # key encoder [backbone, neck]
        self.encoder_k = nn.Sequential(builder.build_backbone(backbone), builder.build_neck(neck))

        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda = loss_lambda
        
        # stiamo creando la coda per le feature globali per la global contrastive loss
        # tensore di dimensione [feat_dim, queue_len]
        # ogni colonna contiene una feature globale k
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # stiamo creando la coda per le feature globali per la dense contrastive loss 
        # tensore di dimensione [feat_dim, queue_len]
        # ogni colonna contiene una feature globale k2
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        
        im_q = img[:, 0, ...].contiguous() # first view
        im_k = img[:, 1, ...].contiguous() # second view
        
        # encoder_q = [backbone, neck], quindi encoder_q[0] = backbone
        # leggere commento in densecl_coco_lmdb_800ep.py 
        # output è una lista, con un elemento, di dimensione [B, C_backbone, H, W] corrispondente all'ultimo stage della rete
        q_b = self.encoder_q[0](im_q)  

        # encoder_q = [backbone, neck], quindi encoder_q[1] = neck
        # 3 output:
        # 1) q: feature globali per ogni immagine del batch, dimensione [B, feat_dim]
        #    utilizzate per la loss contrastiva globale
        # 2) q_grid: feature dense per ogni immagine del batch, dimensione [B, feat_dim, S^2],
        #    con S^2 = H*W, utilizzate per la loss contrastiva densa
        # 3) q2: feature globali per ogni immagine del batch, dimensione [B, feat_dim]
        #    utilizzate per la loss contrastiva densa
        # nota: q e q2 sono diversi – vedere il metodo forward nella classe DenseCLNeck in necks.py
        q, q_grid, q2 = self.encoder_q[1](q_b) 
 
        # prendiamo primo (e unico) tensore della lista q_b
        # reshaping del [B, C, H, W] a [B, C, H*W] = [B, C, S^2]
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        # normalizzazione-L2
        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # todo: aggiungere commento
        with torch.no_grad(): 
            # todo aggiungere commento
            self._momentum_update_key_encoder()  

            # mescola il batch di immagini im_k (non ho capito why, ma ha a che fare con DDP - c'entra poco ora)
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k) 

            # idem come sopra
            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            # idem come sopra
            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # riordiniamo le immagini (non ho capito why, ma ha a che fare con DDP - c'entra poco ora)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # stiamo calcolando i positive logits
        # ovvero, per ogni immagine del batch, calcoliamo il prodotto scalare (einsum)
        # tra la feature globale q e la feature globale k della stessa immagine
        # ottenendo tensore di dimensione Bx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # stiamo calcolando i negative logits
        # ovvero, per ogni immagine del batch, calcoliamo il prodotto scalare
        # tra la feature globale q e tutte le feature globali k della coda
        # ottenendo tensore di dimensione BxK (K = queue_len)
        # ovvero, prodotto matrice-matrice tra [B, C] e [C, K] (da cui la dimensione del tensore)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # algoritmo di similarità (paragrafo 2.4 Dense Correspondence across Views)
        # q_b batch di feature maps del query backbone
        # q_b ha dimensione [B, C, S^2]
        # q_b.permute(0, 2, 1) ha dimensione [B, S^2, C]
        # k_b batch di feature maps del key backbone
        # k_b ha dimensione [B, C, S^2]
        # risultato ha dimensione [B, S^2, S^2]
        # ovvero, per ogni immagine del batch, otteniamo una matrice di similarità S^2xS^2
        # in cui l'elemento (i,j) rappresenta la similarità tra 
        # il vettore di C canali del punto spaziale i della query e 
        # il vettore di C canali del punto spaziale j della key
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)

        # se prendiamo un immagine del batch, otteniamo un "vettore" di lunghezza S^2
        # la posizione i-esima di questo vettore = indice della posizione nella key che è più simile alla posizione i della query
        # dimensione finale [B, S^2]
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]

        # k_grid = dense feature maps del key backbone
        # stiamo riordinando i pixels della dense feature maps della key
        # in base agli indici di similarità calcolati precedentemente
        # in particolare, posizione (i, j) della query ↔ posizione (i, j) della nuova key mappa = match più simile trovato
        # indexed_k_grid ha dimensione [B, C, S^2]
        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) 
        
        # prodotto scalare tra ogni pixel della query e il pixel più simile trovato nella key
        # densecl_sim_q ha dimensione [B, S^2]
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) 

        # stiamo rimodellando densecl_sim_q in modo che abbia dimensione [B*S^2, 1]
        # abbiamo trovato i logits positive per la dense contrastive loss
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) 

        # q_grid ha dimensione [B, C, S^2]
        # lo rimodelliamo in modo che abbia dimensione [B, S^2, C]
        q_grid = q_grid.permute(0, 2, 1)
        # q_grid ha dimensione [B, S^2, C]
        # lo rimodelliamo in modo che abbia dimensione [B*S^2, C]
        q_grid = q_grid.reshape(-1, q_grid.size(2))

        # queue2 contiene le feature globali k2 della key
        # ogni pixel della query e calcola quanto è simile a tutti i pixel negativi memorizzati nella queue2
        # producendo i dense negative logits
        # nota: è un confronto pixel-to-global, e non pixel-to-pixel come per i positive logits 
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid, self.queue2.clone().detach()])

        # qui vengono calcolate le loss (non approfondito)
        loss_single = self.head(l_pos, l_neg)['loss_contra']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss_contra']

        # qui vengono combinate le loss (non approfondito)
        losses = dict()
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda

        # qui vengono aggiornate le code (non approfondito)
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses

    def forward_test(self, img, **kwargs):
        im_q = img.contiguous()
        # compute query features
        #_, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.backbone(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        return None, q_grid, None

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
