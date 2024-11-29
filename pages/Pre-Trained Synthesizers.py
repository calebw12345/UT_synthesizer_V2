# python3 -m venv venv
# source venv/bin/activate

import streamlit as st
from VAE import VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import random

st.set_page_config(layout="wide")
st.sidebar.image('supporting_images/logo_green.png', use_column_width=True)

bins = [
    {"label": "Bin-1: 0% Backwall", "start": 0, "end": 1999, "samples": 2000},
    {"label": "Bin-2: 67.64% to 66.02%", "start": 2000, "end": 2999, "samples": 1000},
    {"label": "Bin-3: 66.02% to 64.40%", "start": 3000, "end": 3999, "samples": 1000},
    {"label": "Bin-4: 64.4% to 62.78%", "start": 4000, "end": 4733, "samples": 734},
    {"label": "Bin-5: 62.78% to 61.17%", "start": 4734, "end": 5220, "samples": 487},
    {"label": "Bin-6: 61.17% to 59.55%", "start": 5221, "end": 5618, "samples": 398},
    {"label": "Bin-7: 59.55% to 57.93%", "start": 5619, "end": 5829, "samples": 211},
    {"label": "Bin-8: 57.93% to 56.31%", "start": 5830, "end": 5949, "samples": 120},
    {"label": "Bin-9: 56.31% to 54.69%", "start": 5950, "end": 6949, "samples": 1000},
    {"label": "Bin-10: 54.69% to 53.07%", "start": 6950, "end": 7949, "samples": 1000},
    {"label": "Bin-11: 53.07% to 51.46%", "start": 7950, "end": 8671, "samples": 722},
    {"label": "Bin-12: 51.46% to 49.84%", "start": 8672, "end": 9332, "samples": 661},
    {"label": "Bin-13: 49.84% to 48.22%", "start": 9333, "end": 10332, "samples": 1000},
    {"label": "Bin-14: 48.22% to 46.60%", "start": 10333, "end": 11148, "samples": 816},
    {"label": "Bin-15: 46.6% to 44.98%", "start": 11149, "end": 11918, "samples": 770},
    {"label": "Bin-16: 44.98% to 43.37%", "start": 11919, "end": 12493, "samples": 575},
    {"label": "Bin-17: 43.37% to 41.75%", "start": 12494, "end": 13147, "samples": 654},
    {"label": "Bin-18: 41.75% to 40.13%", "start": 13148, "end": 13966, "samples": 819},
    {"label": "Bin-19: 40.13% to 38.51%", "start": 13967, "end": 14666, "samples": 700},
    {"label": "Bin-20: 38.51% to 36.89%", "start": 14667, "end": 15313, "samples": 647},
    {"label": "Bin-21: 36.89% to 35.28%", "start": 15314, "end": 15952, "samples": 639},
    {"label": "Bin-22: 35.28% to 33.66%", "start": 15953, "end": 16573, "samples": 621},
    {"label": "Bin-23: 33.66% to 32.04%", "start": 16574, "end": 17099, "samples": 526},
    {"label": "Bin-24: 32.04% to 30.42%", "start": 17100, "end": 17561, "samples": 462},
    {"label": "Bin-25: 30.42% to 28.80%", "start": 17562, "end": 17895, "samples": 334},
    {"label": "Bin-26: 28.80% to 27.18%", "start": 17896, "end": 18137, "samples": 242},
    {"label": "Bin-27: 27.18% to 25.57%", "start": 18138, "end": 18339, "samples": 202},
    {"label": "Bin-28: 25.57% to 23.95%", "start": 18340, "end": 18482, "samples": 143},
    {"label": "Bin-29: 23.95% to 22.33%", "start": 18483, "end": 18669, "samples": 187},
    {"label": "Bin-30: 22.33% to 20.71%", "start": 18670, "end": 18894, "samples": 225},
    {"label": "Bin-31: 20.71% to 19.09%", "start": 18895, "end": 19206, "samples": 312},
    {"label": "Bin-32: 19.09% to 17.48%", "start": 19207, "end": 19515, "samples": 309},
    {"label": "Bin-33: 17.48% to 15.86%", "start": 19516, "end": 19813, "samples": 298},
    {"label": "Bin-34: 15.86% to 14.24%", "start": 19814, "end": 20256, "samples": 443},
    {"label": "Bin-35: 14.24% to 12.62%", "start": 20257, "end": 21036, "samples": 780},
    {"label": "Bin-36: 12.62% to 11.00%", "start": 21037, "end": 21888, "samples": 852},
    {"label": "Bin-37: 11.00% to 9.39%", "start": 21889, "end": 22467, "samples": 579},
    {"label": "Bin-38: 9.39% to 7.77%", "start": 22468, "end": 22973, "samples": 506},
    {"label": "Bin-39: 7.77% to 6.15%", "start": 22974, "end": 23621, "samples": 648},
    {"label": "Bin-40: 6.15% to 4.53%", "start": 23622, "end": 24508, "samples": 887},
    {"label": "Bin-41: 4.53% to 2.91%", "start": 24509, "end": 25508, "samples": 1000},
    {"label": "Bin-42: 2.91% to 1.29%", "start": 25509, "end": 25891, "samples": 383}
]

st.title("Pre-Trained Synthesizers")
ascan_num = 1

st.write("If you do not want to synthesize your own custom datatype, or if you would like to view the level of realism in the A-scans of models that have already been trained, then feel free to interact with our pretrained models below! These models will produce A-scans of corrosion flaw type, and uncorroded backwall. In the instance of both corrosion and backwall, the synthesizers are attempting to produce data from a planar component of aluminum material type, .75 inches (19.05mm) in thickness. There are 42 synthesizers total; one synthesizer for backwall and 41 synthesizers for corrosion of through-wall 66% to 2%. Refer to the image below for an explanation of through-wall. Each corrosion synthesizer will produce data of its corresponding through-wall range. Every corrosion synthesizer produces A-scans of a 1.5% through-wall range.")
st.image('supporting_images/remligfigure.png')
st.write("(Image Above Found on https://www.epri.com/research/products/000000003002031138)")

new_title3 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 32px;"><b>Pre-Trained Synthesizer Instructions</b></p>'
st.markdown(new_title3, unsafe_allow_html=True)
st.write("Below states a step by step guide on how to produce synthetic data:")
st.write("Under 'Synthesizer Options' you will find 1 area to input a number, a button that states 'Change A-scan in view', and two drop down menus.")
st.write("Step 1. Enter a number greater than zero into the area under 'Enter The Number of A-scans You Want to Synthesize (Has to Be Greater Than 0)'. It is recommended that you enter a number greater than 1000, keeping in mind that the larger this number is the more time the synthesizer will take to produce your data.")
st.write("Step 2. Click on the dropdown area under 'Which Model Would You Like to Use to Synthesize Your Data?', there are two options here. The 'Variation Autoencoder' will produce more realistic A-scans, so it is recommeded that you select the 'Variation Autoencoder' option. However, if you are interested in comparing the performance of the Variation Autoencoder to the Generative Adversarial Network feel free to select the 'Generative Adversarial Network' option after you have viewed the A-scans from the Variation Autoencoder.")
st.write("Step 3. Under 'Which Data Type Would You Like to Synthesize?' select the type of data you are interested in viewing. ")
st.write("After a number has been input in 'Enter The Number of A-scans You Want to Synthesize (Has to Be Greater Than 0)', and an option has been selected for both 'Which Model Would You Like to Use to Synthesize Your Data?' and 'Which Data Type Would You Like to Synthesize?', you will see a number of plots appear. The first plot displays 1 synthetic A-scan of the total number of A-scans that you produced, if you want to change which A-scan is being shown in this plot you can click the 'Change A-scan in View' button. Some of the other plots shown may display a large discrepancy between the real and synthetic data if you input to synthesize less than 1000 A-scans due to the fact that the ~1000 A-scans are being displayed in each plot corresponding to real data (which is again the reason for previously recommending to synthesize more than 1000 A-scans).The second plot displays the mean of the synthetic data produced by the generator, compared to the mean of the real data that the synthesizer as attempting to mimic. The third plot displays the standard deviation of the synthetic data produced by the generator, compared to the standard deviation of the real data that the synthesizer as attempting to mimic. The fourth and fifth plots show the distribution of max amplitudes of the first reflection in all the synthetic data that was produced as well as of the real data ths ynthesizer was attempting to mimic. Finally, the last plot shown is a loss plot detailing some specifications of how the synthesizer was trained to mimic the real data.")
st.write("")
st.write("")
new_title8 = '<p style="font-family:sans-serif; color:rgb(255, 255, 255); font-size: 20px;"><b>Enter The Number of A-scans You Want to Synthesize (Has to Be Greater Than 0):</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
totscans = st.text_input("", 1)
totscans = int(totscans)
if st.button('Change A-scan in View'):
  ascan_num = random.randint(1, totscans-1)


def generate_synthetic_data(path,binnumber):
  lblforplot = bins[binnumber]
  lblforplot = str(lblforplot["label"])
  input_dim = 1300  # 1300
  latent_dim = 100  # Latent space dimension
  lr = 1e-5
  batch_size = 64
  epochs = 200
  filepath = path
  model_state = torch.load(filepath)
  vae_model = VAE(input_dim, latent_dim)  # 1300
  vae_model.load_state_dict(model_state)
  vae = vae_model
  with torch.no_grad():
    z = torch.randn(int(totscans), latent_dim)  # Generate 2000 new synthetic rows
    synthetic_data = vae.decode(z).numpy()
    print(synthetic_data)  # The generated rows
  synthetic_datanp = synthetic_data
  synthetic_data = pd.DataFrame(synthetic_data)
  x = synthetic_data.iloc[int(ascan_num)-1,:]
  micros = np.arange(1300)
  micros = micros/100
  fig, ax = plt.subplots(figsize=(5, 5))
  if binnumber == 0:
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(totscans)+" | Type of A-scan:"+lblforplot[9:])
  else:
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(totscans)+" | Type of A-scan: "+lblforplot[6:]+" Through-wall Corrosion")
  ax.set_ylabel("Normalized Amplitude")
  ax.set_xlabel("Microseconds")
  ax.plot(micros,x)
  # Display the plot in Streamlit
  st.pyplot(fig)


  data = np.load('real_data_by_bin/bin'+str(binnumber)+'.npy')
  data = data/np.max(data)
  realmean = np.mean(abs(data), axis=0)
  synmean = np.mean(abs(synthetic_datanp), axis=0)
  side = pd.DataFrame(range(0,len(realmean)))*.01
  side = side.transpose()
  side = side.iloc[0,:]
  side1 = pd.DataFrame(range(0,len(synmean)))*.01
  side1 = side1.transpose()
  side1 = side1.iloc[0,:]
  fig1, ax1 = plt.subplots(figsize=(5, 5))
  ax1.plot(side, realmean)
  ax1.plot(side1, synmean)
  ax1.set_title("Mean of Real vs Synthetic Backwall A-scans")
  ax1.set_xlabel("Time (Microseconds)")
  ax1.set_ylabel("Normalized Amplitude")
  ax1.legend(["Real","Synthetic"])  

  realstd = np.std(abs(data), axis=0)
  synstd = np.std(abs(synthetic_datanp), axis=0)
  fig2, ax2 = plt.subplots(figsize=(5, 5))
  ax2.plot(side, realstd)
  ax2.plot(side1, synstd)
  ax2.set_title("Standard Deviation of Real vsSynthetic Backwall A-scans")
  ax2.set_xlabel("Time (Microseconds)")
  ax2.set_ylabel("Normalized Amplitude")
  ax2.legend(["Real","Synthetic"])

  firstref_amp = pd.DataFrame()
  full_firstref = pd.DataFrame(pd.DataFrame(data).iloc[:,600:700])
  for ascan in range(0,len(full_firstref)):
    ma = pd.DataFrame(full_firstref.iloc[ascan])
    ma = ma.max()
    firstref_amp = pd.concat([firstref_amp,ma])

  firstref_amp_syn = pd.DataFrame()
  full_firstref_syn = pd.DataFrame(pd.DataFrame(synthetic_datanp).iloc[:,600:700])
  for ascan in range(0,len(full_firstref_syn)):
    ma = pd.DataFrame(full_firstref_syn.iloc[ascan])
    ma = ma.max()
    firstref_amp_syn = pd.concat([firstref_amp_syn,ma])
  # fig3, ax3 = plt.subplots(figsize=(5, 5))
  # firstref_amp = firstref_amp[firstref_amp < .2]
  # ax3.hist(firstref_amp,edgecolor='black', linewidth=1.2)
  # ax3.set_title("Distribution of First Reflection Max Amplitudes of Real Data")
  # ax3.set_xlabel("Amplitude")
  # ax3.set_ylabel("Count")
  # ax3.legend(["Real","Synthetic"])

  fig4, ax4 = plt.subplots(figsize=(5, 5))
  ax4.hist(firstref_amp_syn,edgecolor='black', linewidth=1.2)
  ax4.set_title("Distribution of First Reflection Max Amplitudes of Synthetic Data")
  ax4.set_xlabel("Amplitude")
  ax4.set_ylabel("Count")
  st.pyplot(fig1)
  st.pyplot(fig2)
  st.image('VAE/post_clean_metrics/dist'+str(binnumber)+".png")
  # st.pyplot(fig3)
  st.pyplot(fig4)
  

# Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def generate_synthetic_data_gan(path,binnumber):
  lblforplot = bins[binnumber]
  lblforplot = str(lblforplot["label"])
  input_dim = 1300  # 1300
  latent_dim = 100  # Latent space dimension
  lr = 1e-5
  batch_size = 64
  epochs = 200
  filepath = path
  model_state = torch.load(filepath)
  vae_model = Generator(latent_dim, input_dim)
  vae_model.load_state_dict(model_state)
  vae = vae_model
  vae.eval()
  with torch.no_grad():
      noise = torch.randn(totscans, latent_dim)
      synthetic_data = vae(noise).cpu().numpy()
  synthetic_datanp = synthetic_data
  synthetic_data = pd.DataFrame(synthetic_data)
  x = synthetic_data.iloc[int(ascan_num)-1,:]
  micros = np.arange(1300)
  micros = micros/100
  fig, ax = plt.subplots(figsize=(5, 5))
  if binnumber == 0:
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(totscans)+" | Type of A-scan:"+lblforplot[9:])
  else:
    ax.set_title("Synthetic A-scan #"+str(ascan_num)+" of "+str(totscans)+" | Type of A-scan: "+lblforplot[6:]+" Through-wall Corrosion")
  
  ax.set_ylabel("Normalized Amplitude")
  ax.set_xlabel("Microseconds")
  ax.plot(micros,x)
  # Display the plot in Streamlit
  st.pyplot(fig)


  data = np.load('real_data_by_bin/bin'+str(binnumber)+'.npy')
  data = data/np.max(data)
  realmean = np.mean(abs(data), axis=0)
  synmean = np.mean(abs(synthetic_datanp), axis=0)
  side = pd.DataFrame(range(0,len(realmean)))*.01
  side = side.transpose()
  side = side.iloc[0,:]
  side1 = pd.DataFrame(range(0,len(synmean)))*.01
  side1 = side1.transpose()
  side1 = side1.iloc[0,:]
  fig1, ax1 = plt.subplots(figsize=(5, 5))
  ax1.plot(side, realmean)
  ax1.plot(side1, synmean)
  ax1.set_title("Mean of Real vs Synthetic Backwall A-scans")
  ax1.set_xlabel("Time (Microseconds)")
  ax1.set_ylabel("Normalized Amplitude")
  ax1.legend(["Real","Synthetic"])  

  realstd = np.std(abs(data), axis=0)
  synstd = np.std(abs(synthetic_datanp), axis=0)
  fig2, ax2 = plt.subplots(figsize=(5, 5))
  ax2.plot(side, realstd)
  ax2.plot(side1, synstd)
  ax2.set_title("Standard Deviation of Real vsSynthetic Backwall A-scans")
  ax2.set_xlabel("Time (Microseconds)")
  ax2.set_ylabel("Normalized Amplitude")
  ax2.legend(["Real","Synthetic"])

  firstref_amp = pd.DataFrame()
  full_firstref = pd.DataFrame(pd.DataFrame(data).iloc[:,600:700])
  for ascan in range(0,len(full_firstref)):
    ma = pd.DataFrame(full_firstref.iloc[ascan])
    ma = ma.max()
    firstref_amp = pd.concat([firstref_amp,ma])

  firstref_amp_syn = pd.DataFrame()
  full_firstref_syn = pd.DataFrame(pd.DataFrame(synthetic_datanp).iloc[:,600:700])
  for ascan in range(0,len(full_firstref_syn)):
    ma = pd.DataFrame(full_firstref_syn.iloc[ascan])
    ma = ma.max()
    firstref_amp_syn = pd.concat([firstref_amp_syn,ma])
  # fig3, ax3 = plt.subplots(figsize=(5, 5))
  # firstref_amp = firstref_amp[firstref_amp < .2]
  # ax3.hist(firstref_amp,edgecolor='black', linewidth=1.2)
  # ax3.set_title("Distribution of First Reflection Max Amplitudes of Real Data")
  # ax3.set_xlabel("Amplitude")
  # ax3.set_ylabel("Count")
  # ax3.legend(["Real","Synthetic"])
  

  fig4, ax4 = plt.subplots(figsize=(5, 5))
  ax4.hist(firstref_amp_syn,edgecolor='black', linewidth=1.2)
  ax4.set_title("Distribution of First Reflection Max Amplitudes of Synthetic Data")
  ax4.set_xlabel("Amplitude")
  ax4.set_ylabel("Count")
  st.pyplot(fig1)
  st.pyplot(fig2)
  st.image('GAN/post_clean_metrics/dist'+str(binnumber)+".png")
  # st.pyplot(fig3)
  st.pyplot(fig4)
  

b = pd.DataFrame(bins)
b = b["label"]
first = pd.DataFrame(["-"])
first.columns = ["label"]
finlbls = pd.concat([first,b])
finlbls = finlbls.reset_index()
finlbls = finlbls.drop('index',axis=1)

new_title8 = '<p style="font-family:sans-serif; color:rgb(255, 255, 255); font-size: 20px;"><b>Which Model Would You Like to Use to Synthesize Your Data?</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
model_option = st.selectbox(
    "",
    (["-","Variation Autoencoder","Generative Adversarial Network"]),
)

new_title8 = '<p style="font-family:sans-serif; color:rgb(255, 255, 255); font-size: 20px;"><b>Which Data Type Would You Like to Synthesize?</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
option = st.selectbox(
    "",
    (finlbls["label"].to_list()),
)
finlbls = finlbls["label"].to_list()

if str(model_option) == "Generative Adversarial Network":
  for value in finlbls:
    if str(option) == value:
      to_p = finlbls.index(str(option))
      to_p = to_p-1
      if to_p == -1:
         print()
      else:
        path = 'GAN/GAN_pretrained/gen'+str(to_p)+'.pth'
        generate_synthetic_data_gan(path,int(to_p))
        st.image('GAN/GAN_pretrained/lossplot_GAN_bin'+str(to_p)+'.png')



if str(model_option) == "Variation Autoencoder":
  for value in finlbls:
    if str(option) == value:
      to_p = finlbls.index(str(option))
      to_p = to_p-1
      if to_p == -1:
         print()
      else:
        path = 'VAE/VAE_pretrained/backwall_vae_'+str(to_p)+'.pth'
        generate_synthetic_data(path,int(to_p))
        st.image('VAE/VAE_pretrained/lossplot_bin'+str(to_p)+'.png')
        
