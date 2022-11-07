# OrgImpact_noPPI
Measuring an Organization's Impact through Network Connections


ABOUT THE PROJECT: 


This is a copy of the data analtics code for the ASU Unit for Data Science and Analytics' Neptune project, 
but with a created test set of data in order to maintain the protected information of the actual personnel involved. 
The dummy data is yet to be populated, which means the code also needs to change to support the data structure of the dummy data. 

The goal of the project was to measure the impact of a broker organization. Due to the different organization locations having very different processes, 
we decided to create a survey application asking about the organization's business network, ie who works with whom.
We worked hand-in-hand with stakeholders to find out what measurements we could and should use.

The team wrote a web-based survey application, the data was stored in a JSON database, that data was loaded into Python and performed processing and analysis. The team also developed a proof of concept dashboard so stakeholders could have an interactive tool which can provide feedback and about measurements as wellas visualizatin of the network. 

My main focus was the data processing, data analytics, algorithm development and initial data visualization, but I was involved with every aspect of the project. 



ABOUT THE CODE:


"create.py"  will be a file instantiating the data from JSON database, and preforming pre-processing.

"analysis.py" will be the data analysis

"random.py" will be the creation of constrained random graphs and performing the same analysis as the data analysis for comparison 

