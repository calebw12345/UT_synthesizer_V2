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

#Varaiation Autoencoder Class used to train custom synthesizer. NOTE: In the future this will be moved to a backend.
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
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_div

#Trains model and returns synthetic data
def train_model(data,numoption):
    st.write("Training in progress...")
    # Hyperparameters
    input_dim = data.shape[1]  # 1300
    latent_dim = latent_dim  # Latent space dimension
    lr = lr
    batch_size = batch_size
    epochs = epochs

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
            optimizer.zero_grad()
            recon_x, mean, log_var = vae(x)
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

#STREAMLIT PAGE STARTS HERE

st.set_page_config(layout="wide")
st.sidebar.image('supporting_images/logo_green.png', use_column_width=True)

st.title("Custom Synthesizer")

#Page description
st.write("In this page you will be prompted to upload a file of a particular type of UT data that you want to be synthesized. After you upload your data file, enter the hyperparameters that you want your model to use. It is recommended that you conduct your own independent research to determine which hyperparameters are best to obtain the highest accuracy for your specific dataspace. You are welcome to use the default hyperparameters, however the level of realism in the synthetic data is likely to be low.")

# A file uploader
new_title19 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Please Upload The Type of Data That You Want to Synthesize Here (must be in .npy file format):</b></p>'
st.markdown(new_title19, unsafe_allow_html=True)
with st.expander("Upload A NPY File"):
    uploaded_file = st.file_uploader("", type=["npy"])
 
    if uploaded_file is not None:
        data = np.load(uploaded_file)
        synthetic_data = train_model(data,numoption)
    

        # Check error. 
        if not st.session_state.ok_a:
            with st.spinner("Checking for error log..."):
                is_error = False
        
                if is_error:
                    st.dataframe(['log A'])
                else:
                    st.success("A model has been successfully trained, and you have sucessfully produced "+str(len(synthetic_data))+" Synthetic A-scans!")
                    st.session_state.ok_a = True
        else:
            st.success("A model has been successfully trained!")

#Ask for the number of A-scans to synthesize after model is trained. These A-scans will be used for metrics production
new_title8 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Input the Number of A-scans That You Want to Synthesize (must be greater than 1):</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
numoption = st.text_input("", 10,key=4)

#Ask user for basic hyperparameter input
new_title20 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Enter The Number of Epochs That You Want to Train Your Model To:</b></p>'
st.markdown(new_title20, unsafe_allow_html=True)
epochs = st.text_input("", 100,key=0)
new_title21 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Enter Latent Dimension To Use:</b></p>'
st.markdown(new_title21, unsafe_allow_html=True)
latent_dim = st.text_input("", 40,key=1)
new_title22 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Enter Learning Rate To Use (used for both the generator and discriminator):</b></p>'
st.markdown(new_title22, unsafe_allow_html=True)
lr = st.text_input("", 1e-5,key=2)
new_title23 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 20px;"><b>Enter Batch Size To Use:</b></p>'
st.markdown(new_title23, unsafe_allow_html=True)
batch_size = st.text_input("", 64,key=3)
epochs = int(epochs)
latent_dim = int(latent_dim)
lr = float(lr)
batch_size = int(batch_size)


#Display Message
st.write("The data you upload will be used to train a machine learning model to synthesize your data. A random A-scan produced by your synthesizer will be shown after the model has finished training.")

if 'ok_a' not in st.session_state:
    st.session_state.ok_a = False

try:
    if len(synthetic_data) ==1:
        ascan_num =1
    else:
        ascan_num = random.randint(1, len(synthetic_data)-1)
        
    ascan_num = int(ascan_num)
    synthetic_data = pd.DataFrame(synthetic_data)

    x = synthetic_data.iloc[ascan_num,:]
    micros = np.arange(synthetic_data.shape[1])
    micros = micros/100
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(len(synthetic_data)))
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("Microseconds")
    ax.plot(micros,x)
    st.pyplot(fig)
    # Convert DataFrame to CSV
    csv_data = synthetic_data.to_csv(index=False)
    
    # Add download button
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name="custom_synthesized_data.csv",
        mime="text/csv"
    )
except (TypeError,NameError)  as e:
    print()
