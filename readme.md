# Galactic CNN

This is a basic exploration of the CNN structure, it is meant to be a  basic guide/reference to the overall process that I (or anyone who needs to) can quickly reference if need be. This isn't a mathematically heavy guide, there are plenty of those out there that do a wonderful job of that already, so I felt it best to not try to reinvent the wheel. This is however, for anyone who wants a quick refresher on some of the more common features, or want to quickly run a galactic classification CNN

I include:
1. The location of where to find the data. Unforunately even the zipped filesize is too large to upload and host it here for easy download - so anyone following this guide will need to manually download the file itself. The specific file I grabbed is images_gz2.zip
2. The .ipynb file (CNN-cleaning.ipynb) where I clean the original csv and reduce the different classification types of different galaxies from 679 to 3. This is because there are 3 main ones with a ton of subcategories, and there is a measure of uncertainty to each of them. This file condenses them to make the training simpler and more manageable, however there is an explanation of the different subtypes and what each letter symbolizes, so someone could easily change it to incorporate more classification types
3. The original file (zoo2MainSpecz.csv) that CNN-cleaning edits
4. A .ipynb file (Filename-matching.ipynb) that goes through and adds the information necessary to allow the model to find the right image. There is an 'Object ID' column that the two files are merged on, and this adds the "Asset ID" (which is the index of it's location in the folder) to the classification type-reduced csv from earlier. This completes the compilation of all necessary information for the model
5. The filename matching csv (gz2_filename_mapping.csv), the classification type-reduced csv (galaxy_df_cleaned.csv), and the final csv that contains all the information (galaxy_df_final)
6. The model itself in two different formats: .py and .ipynb - the .py file is where I initially built it and ran it as I prefer using VSCode, but I also included the ipython notebook version as that has superior visualization and markdown capabilities. It allows for better formatted explanations essentially, so I decided to include both. The .py file is more of a "plug and chug" version with comments that have minimal explanations, and the .ipynb goes into details about as much as I could think of
7. The .ipynb visualization of the results (CNN-visualization.ipynb) - this is more of a demonstration of how to plot the results than showing off the results of the model itself. I am happy with the results in that they performed better than expected, but they could easily be improved in multiple ways (please see the limitations and discussion sections of the GalacticCNN.ipynb for further explanations)

To use this you should have the following installed:
* Pandas
* Numpy
* OpenCV (cv2)
* sklearn / scikit-learn
* Tensorflow
* Keras
* Keras_tuner

I used a Python 3.10.11 conda environemnt and conda installed most things, and my local machine is an Alienware M17x R1 running Ubuntu 20.04. Using other versions may break something in the code if ran under different conditions, so please keep that in mind if something does go wrong. Also, be sure to change the filepath as those are configurations specific to my local machine

Hope you find this helpful!
