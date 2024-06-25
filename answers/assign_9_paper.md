<div style="text-align: center;">

# Extreme Event Detection and Classification using Convolution and CNN

</div>

<div style="text-align: center;">

*Vedanshi Vagehla<sup>1</sup>*
</div>

<div style="text-align: center;">

 *Indian Institute of Science Education and Research Bhopal<sup>1</sup>*
</div>

---
<div style="text-align: center;">

#### **Abstract**

*Testing a CNN classifier to see how well would it could classify extreme events in comparison to ANN doing the classification.*

</div>

---
<div>

## ***1. Introduction***
<section>

<p class="intro" style="text-align: left;">

### **1.1 Problem statement:**
- Extreme weather events, such as hurricanes, floods, wildfires, and heatwaves, pose significant threats to human life, infrastructure, and ecosystems. These events have been increasing in frequency and intensity due to climate change, making their timely and accurate detection more critical than ever. Traditional methods for identifying extreme weather events often rely on manual analysis by meteorologists or heuristic-based algorithms. These methods are not only time-consuming and labor-intensive but also susceptible to human error and may lack the precision needed to handle the complex and dynamic nature of weather patterns.

- With the advent of modern data acquisition technologies, there is an abundance of high-resolution environmental data available from various sources such as satellite imagery, weather radars, ground-based sensors, and remote sensing devices. However, the sheer volume and complexity of this data present significant challenges for effective analysis and real-time processing. Existing machine learning models have shown promise in analyzing such data, but they often struggle with the variability and unpredictability inherent in extreme weather events, leading to suboptimal performance in detection and classification tasks.

- This research seeks to address these challenges by developing a Convolutional Neural Network (CNN) model specifically designed for the detection and classification of extreme weather events. The use of CNNs is motivated by their proven ability to automatically learn hierarchical feature representations from raw data, making them particularly well-suited for image and pattern recognition tasks. By leveraging custom-designed convolutional filters, this research aims to enhance the model's ability to capture critical features indicative of extreme weather conditions.

- This system will not only improve the accuracy and speed of detection but also reduce the reliance on manual analysis, thereby enhancing overall disaster preparedness and response efforts. The primary goal is to create a robust and efficient CNN-based system capable of processing large-scale environmental datasets in real-time, providing accurate and timely alerts for extreme weather events. 
</p>
<p class="aim" style="text-align: right;">

### **1.2 Aim:**
- The aim of this research is to create a CNN-based system that automatically detects and classifies extreme weather events using large-scale environmental data. By focusing tailored model architecture, rigorous training and evaluation, real-time processing capabilities and effective visualization tools, the research seeks to address the limitations of current methods and provide a scale-able, accurate, and efficient solution for extreme event detection. The ultimate goal is to enhance disaster preparedness and response efforts, contributing to improved safety and resilience in the face of increasing extreme weather events driven by climate change.
</p>
</section>
</div>
<div>
<section>

## ***2. Data***
<p class="data">

- the data used is global PERCDR data from 2001 to 2010 which is available at 30 km resolution and is used as test and train data for the model by performing the 80-20 split, it follows a general distribution as follows:

    ![distribution of the  global PERCDR data(2001-2010)](image.png)

### **2.1 Data exploration:**
- the data does not have any nan, inf, or masked values

- the preprocessing in this case is:

    1. Extracting the precipitation values and plotting to see the extreme events for a particular day.

    2. Appplying a custom made filter to compute what would the dimenssions of the final output image would be that would do in the ***fcs*** after applying a pooling layer with 2*2 kernal wiht a stride of 2 twice and also to see what a features of the original image would be hgihlighted.

        ![original image](image-2.png)

        ![convoulted image](image-1.png)

### **2.2 Data pre-processing and splitting:**

- We split the data as 80% testing and 20% training, i.e 8 years on testing and 2 years for training (shuffling the training data to see what accuracy the model might give for shuffeled data.)

- We labelled the data to perform the binary classification for prediciting whether or not a given day witnessed any extreme event(s) and made a heatmap of the exteme event(s) for a particular day and printing the total number of days that witneesed extreme events.

    ![heatmap of extreme events](image-3.png)

</p>

<section>
</div>
<div>

## ***3. Methodologies***
<section>
<p>

### **3.1 convolutional nural networks:**

