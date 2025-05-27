# https://anndata.readthedocs.io/en/stable/tutorials/notebooks/annloader.html

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from anndata.experimental.pytorch import AnnLoader

adata = sc.read_h5ad("/gpfs/scratch/blukacsy/adata.h5ad")
adata.uns['log1p']["base"] = None  # bug fix

adata.X = adata.layers["counts"]

adata.obs['study'] = adata.obs['sample'].astype('category')

adata.obs['cell_type'] = adata.obs['cell_type_edit'].astype('category')

adata.obs['size_factors'] = adata.X.sum(axis=1)

encoder_study = OneHotEncoder(sparse_output=False, dtype=np.float32)
encoder_study.fit(adata.obs['study'].to_numpy()[:, None])  # shape (n_cells,1)

encoder_celltype = LabelEncoder()
encoder_celltype.fit(adata.obs['cell_type'])

encoders = {
    'obs': {
        'study': lambda s: encoder_study.transform(s.to_numpy()[:, None]),
        'cell_type': encoder_celltype.transform
    }
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()
        modules = []
        # Hidden layers
        for in_size, out_size in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.05))
        # Final layer
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, *inputs):
        x_cat = torch.cat(inputs, dim=-1)
        return self.fc(x_cat)

class CVAE(nn.Module):
    """
    A conditional VAE that reconstructs gene counts (x) conditioned on "study"
    and tries to predict "cell_type" from latent space.
    """
    def __init__(self, input_dim, n_conds, n_classes, hidden_dims, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: input = x + batch(one-hot), output = mean & logvar
        self.encoder = MLP(
            input_dim + n_conds,
            hidden_dims,
            2 * latent_dim  # mean + logvar
        )

        # Decoder: input = z + batch(one-hot), output = distribution over x
        self.decoder = MLP(
            latent_dim + n_conds,
            hidden_dims[::-1],  # symmetrical
            input_dim
        )

        # Theta for NB overdispersion per batch
        self.theta = nn.Linear(n_conds, input_dim, bias=False)

        # Classifier from z -> cell_type
        self.classifier = nn.Linear(latent_dim, n_classes)

    def model(self, x, batches, classes, size_factors):
        pyro.module("cvae", self) 
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            # Prior on z ~ N(0, 1)
            z_loc = x.new_zeros((batch_size, self.latent_dim))
            z_scale = x.new_ones((batch_size, self.latent_dim))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # Predict class from z
            class_probs = self.classifier(z).softmax(dim=-1)
            pyro.sample("class",
                        dist.Categorical(probs=class_probs),
                        obs=classes)

            # Decode
            dec_mu = self.decoder(z, batches).softmax(dim=-1)
            dec_mu = dec_mu * size_factors[:, None]

            dec_theta = torch.exp(self.theta(batches))

            # NB parameterization: total_count=theta, logits=log(mu) - log(theta)
            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()

            # NegativeBinomial
            pyro.sample("obs",
                        dist.NegativeBinomial(
                            total_count=dec_theta,
                            logits=logits
                        ).to_event(1),
                        obs=x.int())

    def guide(self, x, batches, classes, size_factors):
        batch_size = x.shape[0]
        with pyro.plate("data", batch_size):
            # Encoder output => z mean & logvar
            z_loc_scale = self.encoder(x, batches)
            z_mu = z_loc_scale[:, :self.latent_dim]
            z_logvar = z_loc_scale[:, self.latent_dim:]
            z_scale = torch.sqrt(torch.exp(z_logvar) + 1e-4)

            pyro.sample("latent", dist.Normal(z_mu, z_scale).to_event(1))

use_cuda = torch.cuda.is_available()

dataloader = AnnLoader(
    adata,
    batch_size=128,
    shuffle=True,
    convert=encoders, 
    use_cuda=use_cuda
)

n_conds = len(adata.obs['study'].cat.categories)
n_classes = len(adata.obs['cell_type'].cat.categories)
input_dim = adata.n_vars
latent_dim = 10
hidden_dims = [128, 128]

cvae = CVAE(
    input_dim=input_dim,
    n_conds=n_conds,
    n_classes=n_classes,
    hidden_dims=hidden_dims,
    latent_dim=latent_dim
)
if use_cuda:
    cvae.cuda()

optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = pyro.infer.SVI(
    cvae.model,
    cvae.guide,
    optimizer,
    loss=pyro.infer.TraceMeanField_ELBO()
)

def train_one_epoch(svi, loader):
    epoch_loss = 0.0
    for batch in loader:
        # batch.X is the count data
        # batch.obs['study'] is the one-hot study
        # batch.obs['cell_type'] is the integer label
        # batch.obs['size_factors'] is float sums
        epoch_loss += svi.step(
            batch.X,
            batch.obs['study'],
            batch.obs['cell_type'],
            batch.obs['size_factors']
        )
    return epoch_loss / len(loader.dataset)

# Train for some epochs
NUM_EPOCHS = 210
for epoch in range(NUM_EPOCHS):
    loss = train_one_epoch(svi, dataloader)
    if epoch % 40 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"[Epoch {epoch:02d}] Mean training ELBO: {loss:.4f}")

# save model
model_checkpoint = {
    "cvae_state": cvae.state_dict(),
    "pyro_param_store": pyro.get_param_store().get_state()
}

torch.save(model_checkpoint, "/gpfs/scratch/blukacsy/anndata_torch_nn_v1.pt")
print("Model saved to anndata_torch_nn_v1.pt")

full_data = dataloader.dataset[:] # loads entire adata
with torch.no_grad():
    z_loc_scale = cvae.encoder(full_data.X, full_data.obs['study'])
    z_means = z_loc_scale[:, :latent_dim]  # shape (n_cells, latent_dim)

adata.obsm['X_cvae'] = z_means.cpu().numpy()

pred = cvae.classifier(z_means).argmax(dim=-1)
true = full_data.obs['cell_type']  # integer-encoded
acc = (pred == true).sum().item() / adata.n_obs
print(f"Classification accuracy in latent space: {acc:.4f}")

# sc.pp.neighbors(adata, use_rep='X_cvae')
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=['study', 'cell_type'])