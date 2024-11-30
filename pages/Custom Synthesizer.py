# python3 -m venv venv
# source venv/bin/activate

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import random

st.set_page_config(layout="wide")
st.sidebar.image('supporting_images/logo_green.png', use_column_width=True)

st.title("Please Upload The Type of Data That You Want to Synthesize Here (must be in .npy file format):")

#Display Message
st.write("The data you upload will be used to train a machine learning model to synthesize your data. A random A-scan produced by your synthesizer will be shown after the model has finished training.")

# The following code below is assuming your data is stored in a numpy array of shape (2000, 1300)
# try: 
#     uploaded_file = st.file_uploader("")
# except (TypeError,NameError)  as e:
#     print()

# Define the Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Output mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            # nn.Sigmoid()  # Output values between 0 and 1
            # nn.Tanh()  # Account for values outside of 0 and 1
        )
    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=10)  # Clamp log_var
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var) + 1e-6  # Add small epsilon for stability
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

# Loss function (VAE Loss: Reconstruction loss + KL divergence)
def loss_function(recon_x, x, mean, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # print("Recon loss for this epoch is "+str(recon_loss))
    # print("Recon var1 (recon_x): "+str(recon_x))
    # print("Recon var2 (x): "+str(x))
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_div

#Trains model and returns synthetic data
def train_model(data,numoption):
    st.write("Training in progress...")
    # Hyperparameters
    input_dim = data.shape[1]  # 1300
    latent_dim = 100  # Latent space dimension
    lr = 1e-5
    batch_size = 64
    epochs = 100

    # Prepare data loader
    dataset = TensorDataset(torch.Tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    vae = VAE(input_dim, latent_dim)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    vae.apply(init_weights)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    vae.train()

    #dataframe to hold loss data
    lossdf_50 = pd.DataFrame()
    done_training = False

    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            x = batch[0]
            # print("x is: "+str(x))
            # print("NaNs in x: ", torch.isnan(x).sum().item())  # Check for NaNs
            # print("Infs in x: ", torch.isinf(x).sum().item())  # Check for infinities
            optimizer.zero_grad()
            recon_x, mean, log_var = vae(x)
            # print("recon_x: "+str(recon_x))
            # print("mean: "+str(mean))
            # print("log_var: "+str(log_var))
            loss = loss_function(recon_x, x, mean, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()
        ltoadd = train_loss / len(dataloader.dataset)
        ltoadd = pd.DataFrame([ltoadd])
        lossdf_50 = pd.concat([lossdf_50,ltoadd])
        st.write(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}')

    # Generating new synthetic rows
    vae.eval()
    done_training = True

    with torch.no_grad():
        z = torch.randn(int(numoption), latent_dim)  # Generate 10 new synthetic rows
        synthetic_data = vae.decode(z).numpy()
    return synthetic_data

if 'ok_a' not in st.session_state:
    st.session_state.ok_a = False
numoption = st.text_input("Input the Number of A-scans That You Want to Synthesize (must be greater than 1):", 10)
# A upload
with st.expander("Upload A NPY File"):
    uploaded_file = st.file_uploader("Upload A NPY", type=["npy"])
 
    if uploaded_file is not None:
        data = np.load(uploaded_file)
        synthetic_data = train_model(data,numoption)
        # st.write("You have sucessfully produced "+str(len(synthetic_data))+" Synthetic A-scans")
    

        # Check error. 
        if not st.session_state.ok_a:
            with st.spinner("Checking for error log..."):
                is_error = False
        
                if is_error:
                    st.dataframe(['log A'])
                else:
                    st.success("A model has been successfully trained, and you have sucessfully produced "+str(len(synthetic_data))+" Synthetic A-scans!")
                    st.session_state.ok_a = True
                    # ascan_num = 1
                    # if st.button('Change A-scan in View'):
                    #     ascan_num = random.randint(1, len(synthetic_data)-1)
                        
                    # ascan_num = int(ascan_num)
                    # synthetic_data = pd.DataFrame(synthetic_data)

                    # x = synthetic_data.iloc[ascan_num,:]
                    # micros = np.arange(1300)
                    # micros = micros/100
                    # fig, ax = plt.subplots(figsize=(5, 5))
                    # ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(len(synthetic_data)))
                    # ax.set_ylabel("Normalized Amplitude")
                    # ax.set_xlabel("Microseconds")
                    # ax.plot(micros,x)
                    # st.pyplot(fig)
        else:
            st.success("A model has been successfully trained!")
            # ascan_num = 1
            # if st.button('Change A-scan in View'):
            #     ascan_num = random.randint(1, len(synthetic_data)-1)
                
            # ascan_num = int(ascan_num)
            # synthetic_data = pd.DataFrame(synthetic_data)

            # x = synthetic_data.iloc[ascan_num,:]
            # micros = np.arange(1300)
            # micros = micros/100
            # fig, ax = plt.subplots(figsize=(5, 5))
            # ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(len(synthetic_data)))
            # ax.set_ylabel("Normalized Amplitude")
            # ax.set_xlabel("Microseconds")
            # ax.plot(micros,x)
            # st.pyplot(fig)
try:
    ascan_num = random.randint(1, len(synthetic_data)-1)
    # if st.button('Change A-scan in View'):
    #     ascan_num = random.randint(1, len(synthetic_data)-1)
        
    ascan_num = int(ascan_num)
    synthetic_data = pd.DataFrame(synthetic_data)

    x = synthetic_data.iloc[ascan_num,:]
    synthetic_data = pd.DataFrame(synthetic_data)
    synthetic_data = synthetic_data.to_csv()
    synthetic_data = pd.DataFrame(synthetic_data)
        st.download_button(
            label="Download DataFrame as CSV",
            data=csv_data,
            file_name="example_dataframe.csv",
            mime="text/csv"
        )
    micros = np.arange(1300)
    micros = micros/100
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(len(synthetic_data)))
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("Microseconds")
    ax.plot(micros,x)
    st.pyplot(fig)
except (TypeError,NameError)  as e:
    print()


# init_file_name = ''
# if uploaded_file != None:
#     if uploaded_file.name[-4:] != ".npy":
#         st.write("Wrong file type was uploaded")
#     if init_file_name != uploaded_file.name:
#         init_file_name = uploaded_file.name
#         # st.write("Numpy")
        
        

# Create a figure and axes

# st.write("You have sucessfully produced "+str(len(synthetic_data))+" Synthetic A-scans")
# ascan_num = 1
# ascan_num = int(ascan_num)
# synthetic_data = pd.DataFrame(synthetic_data)
# if st.button('Change A-scan in View'):
#     ascan_num = random.randint(1, len(synthetic_data)-1)

# x = synthetic_data.iloc[ascan_num,:]
# micros = np.arange(1300)
# micros = micros/100
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(len(synthetic_data)))
# ax.set_ylabel("Normalized Amplitude")
# ax.set_xlabel("Microseconds")
# ax.plot(micros,x)
# st.pyplot(fig)







        # def scanthrough(data):
        # # Create a figure and axes
        #     fig1, ax1 = plt.subplots(figsize=(5, 5))
            
        #     for ascan in range(len(synthetic_data)):
        #         # Plot some data
        #         x1 = synthetic_data.iloc[ascan,:]
        #         ax1.plot(x1)
        #         st.pyplot(fig1)

                
        
        # if st.button('Scroll Through All Synthetic A-scans'):
        #     scanthrough(synthetic_data)
