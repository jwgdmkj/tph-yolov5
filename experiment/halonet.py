import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# relative positional embedding

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head = 64,
        heads = 8
    ):
        '''
        선언 값 : block_size = 8, halo_size = 4, dim_head = 64, heads = 4

        block : input을 분할하여 만들 쿼리 사이즈
        halo : query에 더해 추가적으로 만들 key/value 사이즈. 앞뒤(좌우)로 붙으므로, 정확히 커널(key/value)은 block+2*halo
        '''
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        print('init ---------------------------- ')
        print('dim - ', self.dim)   # 512
        print('heads and dim_heads -  ', self.heads, dim_head)    # 4, 64
        print('inner_dim - ', inner_dim)    # 256 = heads * dim_heads = 헤드수 * 헤드별 차원 = 4*64
        print('halo and block - ', self.halo_size, self.block_size) # 4, 8
        print('init ---------------------------- ')

    def forward(self, x):
        '''
        input : [1, 512, 32, 32]
        q_inp : (32, 32)를 block_size(=8x8)씩 16개로 쪼개, batch와 곱 --> [16, 64, 512]
        kv_inp : halo*2만큼 패딩 해 (40, 40)으로 만들고 stride=8로 (16x16)만큼 총 16개로 쪼개, [16, 256, 512]

        q : [16, 64, 256] by Linear(512, 256)
        k, v : [16, 256, 256] by Linear(512, 256)
        q, k, v : [64, 64, 64] / [64, 256, 64] by 256/h = 64, 즉 head만큼 차원을 쪼개 그 head를 batch부분에 삽입
        qktv : [64, 64, 64]
        output : [1, 512, 32, 32] by
            1) batch에서 head만큼 분할 해 dim과 합침(16, 64, 256)
            2) to_out : nn.Linear(256, 512) --> (16, 64, 512). embed 복구
            3) (32, 32)가 현재 (4, 4)는 batch에, (8,8)은 (64)로 합쳐진 상태. 이를 다시 나눠 height를 8x4, width를 8x4로 복구
            --> (1, 512, 32, 32)
        '''
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values
        # q_inp : [16, 64, 512] by block=8일 때, [32, 32]는 각각 4개씩 8x8로 나뉘어지니 [1x4x4, 8x8, 512]
        print('x shape ', x.shape)
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)
        print('q_inp shape, p1, p2 : ', q_inp.shape, block, block)

        # x = [1, 512, 32, 32]일 떄, padding=4, stride=8, kernel=(block+halo*2)=(8+2*4)=16이면
        # padding: [1, 512, 40, 40] + kernel:16 + stride : 8 --> 0~16/8~24/16~32/24~40으로 가로 4회, 세로 4회 진행됨.
        # 또한, 이 16x16이 각각 512차원이므로, 총 [1, 512x256, 16] = [1, 131072, 16]가 됨.
        # 이를, rearrange 통해 batch와 패치를 묶고, 패치당 256개 픽셀, 각 픽셀은 512차원 임베딩 -> [1x16, 256, 512]
        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        print('kv unfold shape ', kv_inp.shape, 'kernelsize ', block+halo*2, 'block halo ', block, halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)
        print('kv rearr shape ', kv_inp.shape)

        # derive queries, keys, values, here, inner_dim = 256
        # so q = [16, 64, 256] / k, v = [16, 256, 256]
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)
        print('q k v shape ', q.shape, k.shape, v.shape)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))
        print('qkv after map ', q.shape, k.shape, v.shape)

        # scale
        q *= self.scale

        # attention, [64, 64, 256]
        sim = einsum('b i d, b j d -> b i j', q, k)
        print('qkt(=sim) shape ', sim.shape)

        # add relative positional bias, still [64, 64, 256]
        sim += self.rel_pos_emb(q)
        print('sim after pos_emb ', sim.shape)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        print('mask unfold and block halo ', mask.shape, block, halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        print('mask repeat d heads ', mask.shape, b, heads)
        print(mask)
        mask = mask.bool()
        print(mask)

        # This line computes the maximum negative value representable by the data type of the sim tensor.
        # This value is often used in attention mechanisms to mask out certain positions by setting their scores
        # to a very negative value, ensuring they get a near-zero weight after applying the softmax function
        max_neg_value = -torch.finfo(sim.dtype).max

        # https://thought-process-ing.tistory.com/79
        # sim의 바꾸고자 하는 값(mask)를 max_neg_value로 변경
        sim.masked_fill_(mask, max_neg_value)

        # attention
        attn = sim.softmax(dim = -1)

        # aggregate
        out = einsum('b i j, b j d -> b i d', attn, v)
        print('out first shape ', out.shape)

        # merge and combine heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)
        print('out after rearr and to_out ', out.shape)

        # merge blocks back to original feature map
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block),
                        w = (w // block), p1 = block, p2 = block)
        print('final output ', out.shape)
        return out