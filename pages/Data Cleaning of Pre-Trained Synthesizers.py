import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import streamlit as st
st.title("Metrics of Pre-Trained Synthesizers")
st.write("All pre-traind synthesizers in this application required data cleaning practices. Outlier removal was the primary practice. For each synthesizer that was trained, a dynamic outlier threshold (indicated by 'upper threshold' in the plots below) was set to determine which datapoints in the training data needed to be removed prior to training the synthesizer.")
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

b = pd.DataFrame(bins)
b = b["label"]
first = pd.DataFrame(["-"])
first.columns = ["label"]
finlbls = pd.concat([first,b])
finlbls = finlbls.reset_index()
finlbls = finlbls.drop('index',axis=1)

new_title8 = '<p style="font-family:sans-serif; color:rgb(255, 255, 255); font-size: 20px;"><b>Data Type Would You Like to View Detailed Metrics On?</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
option = st.selectbox(
    "",
    (finlbls["label"].to_list()),
)

new_title8 = '<p style="font-family:sans-serif; color:rgb(255, 255, 255); font-size: 20px;"><b>Which Pre-Trained Synthesizer Would You Like to View Detailed Metrics On?</b></p>'
st.markdown(new_title8, unsafe_allow_html=True)
option1 = st.selectbox(
    "",
    ("-","Variation Autoencoder","Generative Adversarial Network"),
)

finlbls = finlbls["label"].to_list()
to_p = finlbls.index(str(option))
to_p = to_p-1
if option1 == "Variation Autoencoder":
    st.image('VAE/pre_clean_metrics/dist'+str(to_p)+'.png')
    st.image('VAE/pre_clean_metrics/mean'+str(to_p)+'.png')
    st.image('VAE/pre_clean_metrics/std'+str(to_p)+'.png')
    st.image('VAE/post_clean_metrics/dist'+str(to_p)+'.png')
    st.image('VAE/post_clean_metrics/mean'+str(to_p)+'.png')
    st.image('VAE/post_clean_metrics/std'+str(to_p)+'.png')
if option1 == "Generative Adversarial Network":
    st.image('GAN/pre_clean_metrics/dist'+str(to_p)+'.png')
    st.image('GAN/pre_clean_metrics/mean'+str(to_p)+'.png')
    st.image('GAN/pre_clean_metrics/std'+str(to_p)+'.png')
    st.image('GAN/post_clean_metrics/dist'+str(to_p)+'.png')
    st.image('GAN/post_clean_metrics/mean'+str(to_p)+'.png')
    st.image('GAN/post_clean_metrics/std'+str(to_p)+'.png')
if option1 == "-":
    print()
