# Brain-Computer-Interface-Movement-Decoding
This repo contains code for a machine learning project to decode if a person would like to select a left vs. right option based on the electroencephalogram (EEG) 
signals from the person's brain. For detailed information about the modeling approach, results, etc., see the included pdf file: `Brain-Computer-Interface-Movement-Decoding.pdf`

## Modeling Approach
We used a support vector machine to classify left vs. right intended movement based on the EEG signals from the subject. 
An SVM was used to handle the high-dimensionality of dataset. Many improvements were made upon a basic SVM, which are discussed in the attached pdf file.

## Dataset
To evaluate the modeling approach, we used two different datasets. The first dataset, which we called "overt", contains data 
from subjects who were actually moving thei left or right hands to select an option. The second dataset, which we called "imagined", contains data where the subjects
are imagining moving their left or right hand, but not actually moving them.
