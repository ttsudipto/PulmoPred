# PulmoPred
Classification of obstructive and non-obstructive pulmonary diseases on the basis of spirometry using 
machine learning (ML)

 Supervised learning classifiers were developed with Support Vector Machine (SVM), Random Forest (RF), 
 Multi-layer Perceptron (MLP), and Naive Bayes (NB) algorithms. The models were trained on spirometry data 
 of patients from Institute of Pulmocare and Research (IPCR), Kolkata, India. The MLP model showed optimal
 performance and is used in the web application.
 
 **Cite as:** 

>Bhattacharjee, S., Saha, B., Bhattacharyya, P., & Saha, S. (2022). Classification of obstructive and 
non-obstructive pulmonary diseases on the basis of spirometry using machine learning techniques. *Journal 
of Computational Science*, 63, 101768.
[https://doi.org/10.1016/j.jocs.2022.101768](https://doi.org/10.1016/j.jocs.2022.101768).

## Using the tool
PulmoPred is available at: http://dibresources.jcbose.ac.in/ssaha4/pulmopred.

To know more about the spirometry features and the methodology, please refer to the 
[About](http://dibresources.jcbose.ac.in/ssaha4/pulmopred/about.html) page. Please refer to the 
[Help](http://dibresources.jcbose.ac.in/ssaha4/pulmopred/help.html) page for understanding the 
inputs and outputs. The dataset used for training and testing the models is available 
[here](http://dibresources.jcbose.ac.in/ssaha4/pulmopred/datasets.php?type=train).

## Development
Python libraries used for developing the ML models :

* numpy (Version-`1.16.6`)
* scikit-learn (Version-`0.20.3`)
* joblib (Version-`0.14.1`)
* scipy (Version-`1.2.3`)
* statistics (Version-`1.0.3.5`)

It is deployed in a Apache HTTP server running Python (version `3.4`).

## Team
* **Sudipto Bhattacharjee** *([ttsudipto@gmail.com](mailto:ttsudipto@gmail.com))*<br/>
  Ph.D. Scholar,<br/>
  Department of Computer Science and Engineering,<br/>
  University of Calcutta, Kolkata, India.<br/>
* **Dr. Banani Saha** *([bsaha_29@yahoo.com](mailto:bsaha_29@yahoo.com))*<br/>
  Associate Professor,<br/>
  Department of Computer Science and Engineering,<br/>
  University of Calcutta, Kolkata, India.
* **Dr. Parthasarathi Bhattacharyya** *([parthachest@yahoo.com](mailto:parthachest@yahoo.com))*<br/>
  Consultant Pulmologist,<br/>
  Institute of Pulmocare and Research,<br/>
  Kolkata, India.
* **Dr. Sudipto Saha** *([ssaha4@jcbose.ac.in](mailto:ssaha4@jcbose.ac.in))*<br/>
  Associate Professor,<br/>
  Division of Bioinformatics,<br/>
  Bose Institute, Kolkata, India.
  
*Please contact Dr. Sudipto Saha regarding any further queries.*
