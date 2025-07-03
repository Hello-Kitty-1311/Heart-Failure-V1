# Heart Failure Clinical Records Clustering

this project uses clinical data about heart failure patients and tries to group them using clustering. the goal is to like, see if we can figure out the different types of heart failure based on their traits and lab results and stuff.

## Why i made this ipynb

i was reading this thing about how heart failure is actually like, a bunch of different types of problems even thought it all leads to the same thing so i thought it would be great idea to try and find those different groups using machine learning. if you can group patients right, maybe doctors can give them more specific treatment instead of just a one-size-fits-all.

## How i made it

*   pandas for data wrangling and data manipulation.
*   matplotlib and seaborn for initial data viz
*   scikit-learn for clustering (kmeans, hierarchical, dbscan), pca and scalier
*   Text analysis using TF-IDF and count vecotrizer
*   streamlit made the app part web view

## Struggles and what i have learned

*   clustering is HARD! tuning parameters and picking the right metrics is confusing, but fun learning!
*   rate limit of free API is pain
*   missing data crashes the whole thing (fixed it with fillna!).
*   the graphs didn't look right at first in colab. got that sorted though!
*   so easy to build web web interface and also realistic to use
*   overall, learned lots about machine learning, data viz, and html reports.

## usage of AI

* Error Lens : finds error in realtime
* Amazon Q Cli : real time code suggestion and explains error
* ChatGPT and Claude : solves bigger porblems