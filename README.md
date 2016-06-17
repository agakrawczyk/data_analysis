## data_analysis

### PCA plot and clustering plots 

Can be used for example for analysis of expression data matrices. 

-Creates numpy array with the  random data - the features of the created data can be changed

-centers and scales the data from the array and then calculates the PCA

-clusters the PCA output using 3 different methods: KMeans, Affinity Propagation and MeanShift

-Plots 4 graphs containing PCA and 3 graphs representing different clustering methods.

-Prints silhouette scores for every clustering method in the title of the graph.

#### Requirements 

- Python 3.5.1.
- Scikit-learn, Numpy, Matplotlib 

#### Instalation 

Git can be used to get the copy of the code and example:

https://github.com/agakrawczyk/data_analysis/blob/master/PCA_plot.py



### Heatmap


Creates a heatmap on the basis of the data set  from http://hadobs.metoffice.com/hadsst3/ containing inf about anomalies
in sea surface temperature.

#### Requirements 

- Python 3.5.1.
- Pandas, Matplotlib, Seaborn 


#### Instalation 

Git can be used to get the copy of the code and example:

https://github.com/agakrawczyk/data_analysis/blob/master/Heat_map.py

https://github.com/agakrawczyk/data_analysis/blob/master/HadSST.3.1.1.0_monthly_globe_ts.txt
