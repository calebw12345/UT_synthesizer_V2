# The UT Data Synthesizer
Hello and welcome to the Ultrasonic Testing 9UT) data synthesizer application!

[View the UT Synthesizer Streamlit App here!](https://utsynthesizerv2.streamlit.app/)

## Introduction

Non-Destructive Evaluation (NDE) is a continuously changing field, but one premise has held strong against the test of time; data scarcity. No matter if you are conducting research for Ultrasonic Testing (UT), Eddycurrent Testing, Visual Testing, Radiographic Testing, or any other type of NDE method, data scarcity is always a prevalent issue. The amount of data you have available will heavily impact the product of your research whether it be training and qualification, developing new inspection methods, creating innovative NDE hardware, or developing cutting edge NDE software. In order to alleviate this industry-wide problem we have created this contribution to the initiative of using machine learning algorithms to synthesize NDE data.

UT data is not easy to come by. There are primarily only 2 ways to acquire UT data; to collect the data on a physical component with an encoding setup, or to acquire data via simulation softwares. Collecting data on a physical sample can be time, cost, and labor intensive. Using simulation sotwares to acquire UT data is time and cost intensive, but the data that is produced by simulation softwares often certain lack realistic features such as material and weld interface noise, coupling induced signal effects, ect.

This application allows you to synthesize any type of UT amplitude scans (A-scans) that you desire a larger quantity of unique data for, along a number of post processing and visualiztion tools that can be used to examine the synthetic data's level of realism. There are also pre-built synthesizers in this application that are able to produce amplitude scans data of flaw type corrosion and uncorroded backwall of an aluminum component .75 inches (19.05mm) in thickness. This application supplies over 40 models that can synthesize corrosion of through-wall ranging from 66% to 2%, as well uncorroded material.

## Data Operation/Abstraction Design
The training data used for each machine learning model was sorced from https://www.epri.com/research/products/000000003002031138, which provided a fairly clean dataset as is. In this dataset are 42 bins of data, each corresponding to a particular section of through-wall. Outlier detection and removal was performed on each bin in an attempt to improve model performance. Specifically, all A-scans in which the first reflection had an amplitude greater than 80% of the total data in a particular bin was removed. More information on the process used for outlier detection, and the impact that outlier removal had on the data can be found within the application itself.

## Future Work
For this application, we have focused on Ultrasonic Testing (UT) data, however in the future other types of NDE data will be the subject of investigation. We are also working on further optimizing each model framework to decrease loss and increase the level of realism that each are able to synthesize. Another short term goal is to aggregate synthetic A-scans and implment common nde visualizations such as B-scans, C-scans, ect.

Please reach out with any comments, questions, or feedback! Also, if you would like to collaborate feel free to contact me via linkedin!
