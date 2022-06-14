import audobject
import torch
import torch.nn.functional as F

from models import (
    ConvBlock,
    init_bn,
    init_layer,
    Cnn14
)


class GradReverse(torch.autograd.function.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

    def grad_reverse(x, lambd=1):
        return GradReverse.apply(x, lambd)


class EmbedConvBlock(ConvBlock):
    def __init__(self, embedding_dim: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.out_channels),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(self.out_channels)
        )

    def forward(self, x, **kwargs):
        embedding = x[1]
        projected_embedding = self.projection(embedding).unsqueeze(-1).unsqueeze(-1)
        x = super().forward(x[0], **kwargs) + projected_embedding
        x = F.dropout2d(x, 0.2, self.training)
        return x


class FixedEmbeddingCnn14(torch.nn.Module, audobject.Object):
    r"""Cnn14 model architecture.

    Args:
        output_dim: number of output classes to be used
        in_channels: number of input channels
    """

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int = 2048,
        in_channels: int = 1,
        use_sigmoid: bool = False
    ):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.use_sigmoid = use_sigmoid
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=in_channels, out_channels=64)
        self.conv_block2 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=64, out_channels=128)
        self.conv_block3 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=128, out_channels=256)
        self.conv_block4 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=256, out_channels=512)
        self.conv_block5 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=512, out_channels=1024)
        self.conv_block6 = EmbedConvBlock(embedding_dim=embedding_dim, in_channels=1024, out_channels=2048)

        self.fc1 = torch.nn.Linear(2048, 2048, bias=True)
        self.out = torch.nn.Linear(2048, self.output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def get_embedding(self, x, embedding):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1((x, embedding), pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2((x, embedding), pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3((x, embedding), pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4((x, embedding), pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5((x, embedding), pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6((x, embedding), pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        return self.clipwise_path(x)

    def forward(self, x):
        embedding = x[1]
        x = x[0]

        main_embedding = self.get_embedding(x, embedding)

        x = F.relu_(self.fc1(main_embedding))
        x = self.out(x)
        if self.use_sigmoid:
            x = (torch.sigmoid(x) + 1) * 7
        return x, main_embedding


class AttentionFusionCnn14(Cnn14):
    def __init__(self, embedding_dim, norm: str = None, residual: bool = True, softmax: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2048),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(self.out_channels)
        )
        self.norm = norm
        self.residual = residual
        if self.norm == 'LayerNorm':
            self.pre_norm = torch.nn.LayerNorm(2048)
            self.post_norm = torch.nn.LayerNorm(2048)
        if self.norm == 'BatchNorm':
            self.pre_norm = torch.nn.BatchNorm1d(2048)
            self.post_norm = torch.nn.BatchNorm1d(2048)
        self.softmax = softmax
    def forward(self, x):
        embedding = x[1]
        x = x[0]

        main_embedding = self.get_embedding(x)
        if self.norm is not None:
            main_embedding = self.pre_norm(main_embedding)
            # print(main_embedding.abs().max())

        embedding = self.projection(embedding)
        if self.softmax:
            embedding = torch.softmax(embedding, 1)
        x = embedding * main_embedding
        if self.norm is not None:
            # print(x.abs().max())
            x = self.post_norm(x)
            # print(x.abs().max())
        if self.residual:  # residual connection for stability
            x = main_embedding + x
        # print((x-main_embedding).abs().max())
        # print()
        # x = embedding * main_embedding

        x = F.relu_(self.fc1(x))
        x = self.out(x)
        if self.sigmoid_output:
            x = (torch.sigmoid(x) + 1) * 7
        return x, main_embedding


class ExemplarNetwork(torch.nn.Module):
    def __init__(self, 
        main_net,
        subnet, 
        adversarial_exemplars: bool = False, 
        auxil_conditions: int = None,
        main_conditions: int = None,
        ignore_auxil: bool = False
    ):
        super().__init__()
        self.main_net = main_net
        self.subnet = subnet
        self.adversarial_exemplars = adversarial_exemplars
        self.grad_reverse = GradReverse.grad_reverse
        self.lambd = -1
        self.output_dim = self.main_net.output_dim
        self.ignore_auxil = ignore_auxil
        self.exemplar_out = torch.nn.Sequential(
            torch.nn.Linear(self.main_net.embedding_dim, self.main_net.embedding_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.main_net.embedding_dim, self.main_net.output_dim, bias=True)
        )
        self.auxil_conditions = auxil_conditions
        if self.auxil_conditions is not None:
            self.condition_out = torch.nn.Sequential(
                torch.nn.Linear(self.main_net.embedding_dim, self.main_net.embedding_dim, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(self.main_net.embedding_dim, self.auxil_conditions, bias=True)
            )
        self.main_conditions = main_conditions
        if self.main_conditions is not None:
            self.main_condition_out = torch.nn.Sequential(
                torch.nn.Linear(2048, 2048, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, self.main_conditions, bias=True)
            )

    def forward(self, input):
        signal = input[0]
        exemplars = input[1]
        
        # simply average exemplars
        if not self.ignore_auxil:
            bs = exemplars.shape[0]
            num_exemplars = exemplars.shape[1]
            exemplars = exemplars.view(-1, 1, exemplars.shape[2], exemplars.shape[3])
            embeddings = self.subnet(exemplars).squeeze(-1).squeeze(-1)
            embeddings = embeddings.view(bs, num_exemplars, -1)
            main_input = (signal, embeddings.mean(1))
        else:
            main_input = signal

        output, main_embeddings = self.main_net(main_input)

        if self.training:
            if not self.ignore_auxil:
                embeddings = embeddings.view(bs * num_exemplars, -1)
                if self.auxil_conditions is not None:
                    # running through before inverse layer kicks in...
                    condition_output = self.condition_out(embeddings)
                if self.adversarial_exemplars:
                    embeddings = self.grad_reverse(embeddings, self.lambd)
                exemplar_output = self.exemplar_out(embeddings)

                if self.auxil_conditions is not None:
                    exemplar_output = (exemplar_output, condition_output)
            else:
                exemplar_output = None
            
            if self.main_conditions is not None:
                # Not sure if using the same grad_inverse will cause problems...
                # TODO: Need to re-introduce grad_reverse once it is working
                output = (output, self.main_condition_out(self.grad_reverse(main_embeddings, self.lambd)))

            return output, exemplar_output
        else:
            return output