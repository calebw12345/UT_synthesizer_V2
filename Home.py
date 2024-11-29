import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import time
# Custom HTML/CSS for the banner
# custom_html = """
# <div class="banner">
#     <img src="https://brand.charlotte.edu/wp-content/uploads/sites/183/2023/04/8716-01-Charlotte-Master-File-v7_1.png" alt="Banner Image">
# </div>
# <style>
#     .banner {
#         width: 35%;
#         height: 600px;
#         overflow: visible;
#     }
#     .banner img {
#         width: 100%;
#         object-fit: fill;
#     }
# </style>
# """
# # Display the custom HTML
# st.components.v1.html(custom_html)

# st.set_page_config(layout="wide")
st.image("supporting_images/logo.png")

# Sidebar content
# st.sidebar.header("Sidebar Title")
# st.sidebar.subheader("Subheading")
# st.sidebar.text("Produced by UNCC")

# Main content
st.title("Welcome to the Ultrasonic Testing Data Synthesizer!")
st.write("Non-Destructive Evaluation (NDE) is a continuously changing field, but one premise has held strong against the test of time; data scarcity. No matter if you are conducting research for Ultrasonic Testing, Eddycurrent Testing, Visual Testing, Radiographic Testing, or any other type of NDE method, data scarcity is always a prevalent issue. The amount of data you have available wll heavily impact the product of your research whether it be training and qualification, developing new inspection methods, creating innovative NDE hardware, or developing cutting edge NDE software. In order to alleviate this industry-wide problem we have created this contribution to the initiative of using machine learning algorithms to synthesize NDE data. For this application, we have focused on Ultrasonic Testing (UT) data, however in the future other types of NDE data will be the subject of future investigation.")
new_title = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 28px;">What is the Benefit of Synthesizing UT Data?</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.write("UT data is not easy to come by. There are primarily only 2 ways to acquire UT data; to collect the data on a physical component with an encoding setup, or to acquire data via simulation softwares. Collecting data on a physical sample can be time, cost, and labor intensive. Using simulation sotwares to acquire UT data is time and cost intensive, but the data that is produced by simulation softwares often certain lack realistic features such as material and weld interface noise, coupling induced signal effects, ect.")
new_title2 = '<p style="font-family:sans-serif; color:rgb(0, 153, 0); font-size: 28px;">Capability of the UT Data Synthesizer</p>'
st.markdown(new_title2, unsafe_allow_html=True)
st.write("This application allows you to synthesize any type of UT amplitude scans that you desire a larger quantity of unique data for, along a number of post processing and visualiztion tools that can be used to examine the synthetic data's level of realism. There are also pre-built synthesizers in this application that are able to produce amplitude scans data of flaw type corrosion and uncorroded backwall of an aluminum component .75 inches (19.05mm) in thickness. This application supplies over 40 models that can synthesize corrosion of through-wall ranging from 66% to 2%, as well uncorroded material.")
st.sidebar.image('supporting_images\logo_green.png', use_column_width=True)
st.write("Interact with the tabs on the left side of the screen to explore the functionalities of this application.")









