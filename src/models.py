import torch
import torch_geometric
import numpy as np
import pytorch_lightning as L


class GATNN_attpool(torch.nn.Module):
    # The default GATNN_attpool network.
    # The architecture is optimized for the IPA spectra with 300 meV broadening
    def __init__(self, dropout_frac=0.0):
        super().__init__()
        self.mlp_init_node = torch_geometric.nn.MLP(
            [13, 48, 48], dropout=dropout_frac, act="relu"
        )
        self.gat1 = torch_geometric.nn.GATv2Conv(
            48,
            48,
            heads=4,
            edge_dim=51,
            concat=True,
            dropout=dropout_frac,
            add_self_loops=False,
        )
        self.gat2 = torch_geometric.nn.GATv2Conv(
            192,
            96,
            heads=4,
            edge_dim=51,
            concat=True,
            dropout=dropout_frac,
            add_self_loops=False,
        )

        self.mlp_att = torch_geometric.nn.MLP([384, 384], act="relu")
        self.mlp1 = torch_geometric.nn.MLP(
            [384, 1024, 1024, 1024, 2001], act="relu")

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        x = torch_geometric.nn.pool.global_add_pool(x * att, graph.batch)
        x = self.mlp1(x)
        x = torch.nn.functional.leaky_relu(x)
        return x

    def latents(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        x = torch_geometric.nn.pool.global_add_pool(x * att, graph.batch)
        return x

    def mp1(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat1(x, edge_index, edge_attr)
        return x

    def mp2(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        return x

    def attmp2(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        return (x*att)


class GATNN_attpool_auto(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        init = torch_geometric.nn.Sequential(
            "x", [(torch_geometric.nn.MLP([13] + params[0], act="relu"), "x -> x")]
        )

        pool = torch_geometric.nn.Sequential(
            "x",
            [
                (
                    torch_geometric.nn.MLP(
                        [params[-2][0] * params[-2][1],
                            params[-2][0] * params[-2][1]],
                        act="relu",
                    ),
                    "x -> att",
                )
            ],
        )
        out = torch_geometric.nn.Sequential(
            "x",
            [
                (
                    torch_geometric.nn.MLP(
                        [params[-2][0] * params[-2][1]] + params[-1] + [2001], act="relu"
                    ),
                    "x -> x",
                )
            ],
        )

        gat_list = []
        for idx, gat in enumerate(params[1:-1]):
            if idx == 0:
                gat_list.append(
                    (
                        torch_geometric.nn.GATv2Conv(
                            params[0][-1],
                            gat[0],
                            heads=gat[1],
                            edge_dim=51,
                            concat=True,
                            add_self_loops=False,
                        ),
                        "x, edge_index, edge_attr -> x",
                    )
                )
            else:
                gat_list.append(
                    (
                        torch_geometric.nn.GATv2Conv(
                            params[idx][0] * params[idx][1],
                            gat[0],
                            heads=gat[1],
                            edge_dim=51,
                            concat=True,
                            add_self_loops=False,
                        ),
                        "x, edge_index, edge_attr -> x",
                    )
                )
        gats = torch_geometric.nn.Sequential(
            "x, edge_index, edge_attr", gat_list)
        self.mlp_init_node = init
        self.gat = gats
        self.mlp_att = pool
        self.mlp1 = out

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        x = torch_geometric.nn.pool.global_add_pool(x * att, graph.batch)
        x = self.mlp1(x)
        x = torch.nn.functional.leaky_relu(x)
        return x

    def latents(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        x = torch_geometric.nn.pool.global_add_pool(x * att, graph.batch)
        return x

    def attmp(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        x = self.gat(x, edge_index, edge_attr)
        if graph.batch == None:
            batch = torch.tensor(np.zeros(len(graph.x)),
                                 dtype=torch.int64).cuda()
        else:
            batch = graph.batch
        att = self.mlp_att(x)
        att = torch_geometric.utils.softmax(att, index=batch)
        return (x*att)

    def lat_before_pool(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        x = self.mlp_init_node(x)
        return x


class LitGatNN(L.LightningModule):
    def __init__(self, lr=1e-5, decay=0, params=[[48, 48], [48, 4], [96, 4], [1024, 1024, 1024]]):
        super().__init__()
        self.gatnn = GATNN_attpool_auto(
            params)
        self.lr = lr
        self.decay = decay

    def training_step(self, batch, batch_idx):
        inputs = batch
        target = batch.y.flatten()
        output = self(inputs).flatten()
        loss = torch.nn.functional.l1_loss(output, target)
        self.log("train", loss, batch_size=len(batch), on_epoch=True)
        # self.logger.experiment.add_scalars("loss",{"train": loss}, self.current_epoch,)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        target = batch.y.flatten()
        output = self(inputs).flatten()
        loss = torch.nn.functional.l1_loss(output, target)
        self.log("valid", loss, on_epoch=True, batch_size=len(batch))
        # self.logger.experiment.add_scalars("loss",{"valid": loss}, self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.decay)
        return optimizer

    def forward(self, inputs):
        return self.gatnn(inputs).flatten()


class LitGatNN_pre(LitGatNN):
    def __init__(self, weight_path, lr=1e-5, decay=0):
        super().__init__(lr, decay)
        self.gatnn = GATNN_attpool()
        self.gatnn.load_state_dict(
            torch.load(weight_path, weights_only=True))
