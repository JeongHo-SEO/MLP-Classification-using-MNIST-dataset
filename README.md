- 2024년 가을학기 DGIST 기계공학트랙의 [인공지능개론] 과제물을 업로드합니다.
- MNIST dataset을 활용하여 MLP를 사용한 분류 모델을 만드는 과제물입니다.
---
### Goals
- This repository covers the assignments for **Introduction to Artificial Intelligence**, lectured by *professor Inkyu Moon*.
- The goal is to create **MLP model to classify the numbers in the MNIST dataset**, using the ***handwriting images and data augmentation***.


### In the First project,
- I used the full-MNIST training dataset for the training. (60,000 samples)
- Then I used *my own hand-writing dataset* for the testing. (30 samples) More specifically,
    - Firstly, I wrote the numbers 0 to 9 using my apple-pencil on my iPad, and then captured. (10 samples)
    - Secondly, I wrote the numbers 0 to 9 using my ball-point pen on the A4 paper, and then took the pictures with my cell phone. (10 samples)
    - Lastly, I wrote the numbers 0 to 9 on the MS words, and then captured. (10 samples)
- Furthermore, I changed the hyper-parameters (learning rate and epochs) and checked the performance.


 ### In the Second Project,
- I used the Augmented full-MNIST training dataset for the training. (180,000 samples) More specifically,
    - I used rotation with angle $$-\theta$$, $$+\theta$$ for all images. As a result, I got more training data: 60,000 $$\rightarrow$$ 180,000 samples. ($$\times 3$$)
- Then I used the full-MNIST testing dataset for the testing. (10,000 samples)
- Furthermore, I changed the hyper-parameters (rotation angle and epochs) and checked the performance.


### Datasets
- For this assignments, I used two version of datasets: small and full version,
    - small version: 100 train samples and 10 test samples.
        - Indeed, the small dataset were used only for the programming.
    - full version: 60,000 train samples and 10,000 test samples.

- The datasets are not included in this repository due to size restrictions.
- Please download from the below links:
    - [small_train](http://www.pjreddie.com/media/files/mnist_train.csv)
    - [small_test](http://www.pjreddie.com/media/files/mnist_test.csv)
    - [full_train and test](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/tree/master/mnist_dataset)

- If the links are not available, you can download the **[Original MNIST website](http://yann.lecun.com/exdb/mnist/)** and convert to the **.csv** files.


### The directories are below:
```
Root/
│
├─ MNIST_dataset_csv/                # CSV-format MNIST datasets (not included in repo) - you have to modify the name of dataset like below.
│   ├─ 10_small_mnist_test.csv
│   ├─ 100_small_mnist_train.csv
│   ├─ 10,000_full_mnist_test.csv
│   └─ 60,000_full_mnist_train.csv
│
├─ Project_1/                        # Project 1: Handwritten digit recognition as the testing dataset.
│   ├─ handwriting_images/           # Input images used for experiments - You can add or delete handwriting images here.
│   │   ├─ 1_Tablet_capture_image/
│   │   ├─ 2_hand_writing_image_phone_scan/
│   │   └─ 3_MS_word_capture_image/
│   ├─ Project_1_result_graph.png    # Visualization of the model performance
│   ├─ Project_1_result_print.png    # Output of the Project_1.py
│   └─ Project_1.py                  # Main script for Project 1
│
├─ Project_2/                        # Project 2: training dataset augmentation using rotation
│   ├─ Project_2_result_graph.png
│   ├─ Project_2_result_print.png
│   └─ Project_2.py
│
├─ Project_report.docx               # Project report (editable version)
└─ Project_report.pdf                # Project report (non-editiable version)
```
