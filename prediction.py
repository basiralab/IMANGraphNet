import os.path as osp
import numpy
import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm
import argparse
from torch.distributions import normal, kl
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder, ARGVA
from torch_geometric.utils import train_test_split_edges
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from losses import*
from model import*
from preprocess import*
from centrality import *
from config import N_TARGET_NODES_F, N_SOURCE_NODES_F,N_TARGET_NODES,N_SOURCE_NODES, N_EPOCHS
warnings.filterwarnings("ignore")
#  GAN
aligner = Aligner()
generator = Generator()
discriminator = Discriminator()
# Losses
adversarial_loss1 = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# send 1st GAN to GPU
aligner.to(device)
generator.to(device)
discriminator.to(device)
adversarial_loss1.to(device)
l1_loss.to(device)

Aligner_optimizer = torch.optim.AdamW(aligner.parameters(), lr=0.025, betas=(0.5, 0.999))
generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.025, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))
def IMANGraphNet (X_train_source, X_test_source, X_train_target, X_test_target):

    X_casted_train_source = cast_data_vector_RH(X_train_source)
    X_casted_test_source = cast_data_vector_RH(X_test_source)
    X_casted_train_target = cast_data_vector_FC(X_train_target)
    X_casted_test_target = cast_data_vector_FC(X_test_target)

    aligner.train()
    generator.train()
    discriminator.train()

    nbre_epochs = N_EPOCHS
    for epochs in range(nbre_epochs):
        # Train Generator
        with torch.autograd.set_detect_anomaly(True):
            Al_losses = []


            Ge_losses = []
            losses_discriminator = []

            i = 0
            for data_source, data_target in zip(X_casted_train_source, X_casted_train_target):
                # print(i)
                targett = data_target.edge_attr.view(N_TARGET_NODES, N_TARGET_NODES)
                # ************    Domain alignment    ************
                A_output = aligner(data_source)
                A_casted = convert_generated_to_graph_Al(A_output)
                A_casted = A_casted[0]

                target = data_target.edge_attr.view(N_TARGET_NODES, N_TARGET_NODES).detach().cpu().clone().numpy()
                target_mean = np.mean(target)
                target_std = np.std(target)

                d_target = torch.normal(target_mean, target_std, size=(1, N_SOURCE_NODES_F))
                dd_target = cast_data_vector_RH(d_target)
                dd_target = dd_target[0]
                target_d = dd_target.edge_attr.view(N_SOURCE_NODES, N_SOURCE_NODES)

                kl_loss = Alignment_loss(target_d, A_output)

                Al_losses.append(kl_loss)

                # ************     Super-resolution    ************
                G_output = generator(A_casted)  # 35 x 35
                # print("G_output: ", G_output.shape)
                G_output_reshaped = (G_output.view(1, N_TARGET_NODES, N_TARGET_NODES, 1).type(torch.FloatTensor)).detach()
                G_output_casted = convert_generated_to_graph(G_output_reshaped)
                G_output_casted = G_output_casted[0]
                torch.cuda.empty_cache()

                Gg_loss = GT_loss(targett, G_output)
                torch.cuda.empty_cache()
                D_real = discriminator(data_target)
                D_fake = discriminator(G_output_casted)
                torch.cuda.empty_cache()
                G_adversarial = adversarial_loss(D_fake, (torch.ones_like(D_fake, requires_grad=False)))
                G_loss = G_adversarial + Gg_loss
                Ge_losses.append(G_loss)

                D_real_loss = adversarial_loss(D_real, (torch.ones_like(D_real, requires_grad=False)))
                # torch.cuda.empty_cache()
                D_fake_loss = adversarial_loss(D_fake.detach(), torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                # torch.cuda.empty_cache()
                losses_discriminator.append(D_loss)
                i += 1

            # torch.cuda.empty_cache()

            generator_optimizer.zero_grad()
            Ge_losses = torch.mean(torch.stack(Ge_losses))
            Ge_losses.backward(retain_graph=True)
            generator_optimizer.step()

            Aligner_optimizer.zero_grad()
            Al_losses = torch.mean(torch.stack(Al_losses))
            Al_losses.backward(retain_graph=True)
            Aligner_optimizer.step()


            discriminator_optimizer.zero_grad()
            losses_discriminator = torch.mean(torch.stack(losses_discriminator))
            losses_discriminator.backward(retain_graph=True)
            discriminator_optimizer.step()

        print("[Epoch: %d]| [Al loss: %f]| [Ge loss: %f]| [D loss: %f]" % (epochs, Al_losses, Ge_losses, losses_discriminator))

    torch.save(aligner.state_dict(), "./weight" + "aligner_fold" + "_" + ".model")
    torch.save(generator.state_dict(), "./weight" + "generator_fold" + "_" + ".model")

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    # #     ######################################### TESTING PART #########################################
    restore_aligner = "./weight" + "aligner_fold" + "_" + ".model"
    restore_generator = "./weight" + "generator_fold" + "_" + ".model"

    aligner.load_state_dict(torch.load(restore_aligner))
    generator.load_state_dict(torch.load(restore_generator))

    aligner.eval()
    generator.eval()

    i = 0
    predicted_test_graphs = []
    losses_test = []
    eigenvector_losses_test = []
    l1_tests = []
    Closeness_test = []
    Eigenvector_test = []
    for data_source, data_target in zip(X_casted_test_source, X_casted_test_target):
        # print(i)
        data_source_test = data_source.x.view(N_SOURCE_NODES, N_SOURCE_NODES)
        data_target_test = data_target.x.view(N_TARGET_NODES, N_TARGET_NODES)


        A_test = aligner(data_source)
        A_test_casted = convert_generated_to_graph_Al(A_test)
        A_test_casted = A_test_casted[0]
        data_target = data_target_test.detach().cpu().clone().numpy()
        # ************     Super-resolution    ************
        G_output_test = generator(A_test_casted)  # 35 x35
        G_output_test_casted = convert_generated_to_graph(G_output_test)
        G_output_test_casted = G_output_test_casted[0]
        torch.cuda.empty_cache()

        L1_test = l1_loss(data_target_test, G_output_test)
        # fold= 1
        target_test = data_target_test.detach().cpu().clone().numpy()
        predicted_test = G_output_test.detach().cpu().clone().numpy()
        source_test = data_source_test.detach().cpu().clone().numpy()

        torch.cuda.empty_cache()
        fake_topology_test = torch.tensor(topological_measures(predicted_test))
        real_topology_test = torch.tensor(topological_measures(target_test))

        eigenvector_test = (l1_loss(fake_topology_test[2], real_topology_test[2]))


        l1_tests.append(L1_test.detach().cpu().numpy())
        Eigenvector_test.append(eigenvector_test.detach().cpu().numpy())



    mean_l1 = np.mean(l1_tests)
    mean_eigenvector = np.mean(Eigenvector_test)

    # print("Mean L1 Loss Test: ", fold_mean_l1_loss)
    # print()

    losses_test.append(mean_l1)
    eigenvector_losses_test.append(mean_eigenvector)

    # fold += 1
    return (predicted_test, data_target, source_test, losses_test, eigenvector_losses_test)



