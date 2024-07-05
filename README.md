# Understanding the link between functional profiles and intelligence through dimensionality reduction and graph analysis
Francesco Alberti, Arianna Menardi, Daniel S. Margulies, Antonino Vallesi


This repository contains the scripts used to perform the analyses of a study investigating the relationdship between interindividual variability of intelligence and the functional organization of the cerebral cortex.\
[Read the preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.12.536421v1)

> **Abstract**\
There is a growing interest in neuroscience for how individual-specific structural and functional features of the cortex relate to cognitive traits. This work builds on previous research which, by using classical high-dimensional approaches, has proven that the interindividual variability of functional connectivity (FC) profiles reflects differences in fluid intelligence. To provide an additional perspective into this relationship, the present study uses a recent framework for investigating cortical organization: functional gradients. This approach places local connectivity profiles within a common low-dimensional space whose axes are functionally interpretable dimensions. Specifically, this study uses a data-driven approach to model the association between FC variability and interindividual differences in intelligence. For one of these loci, in the right ventral-lateral prefrontal cortex (vlPFC), we describe an association between fluid intelligence and the relative functional distance of this area from sensory and high-cognition systems. Furthermore, the topological properties of this region indicate that, with decreasing functional affinity with high-cognition systems, vlPFC functional connections are more evenly distributed across all networks. Participating in multiple functional networks may reflect a better ability to coordinate sensory and high-order cognitive systems.

![image](https://github.com/alberti-f/gradient-dispersion/blob/main/methods_figure.png)\
**Figure 1.** *Visual summary of the analyses. The analyses are composed of two streamlines. First (gray box), we concatenated individual resting-state time series and used GCCA to project them into a common latent space. Within this space we measured vertex-wise dispersion maps that were then thresholded to identify clusters of maximum variability. Then (outside the gray box), we parcellated the original time series using an atlas to which the variability clusters had been added. We built a correlation matrix of regional time series and thresholded it to the top 10% of connections to obtain a graph of functional connectivity. Graph topology was then analyzed to characterize the role of different regions within it. BOLD, blood-oxygen-level-dependent; GCCA, generalized canonical correlation; FC, functional connectivity.*

## Data
* **Human Connectome Project**\
The study uses data ftom the [HCP dataset](https://www.humanconnectome.org/) and the scripts assume the HCP data directory structure. The necessary data are:
    * 4 resting-state fMRI runs
    * Unrestricted behavioral data
    * Restricted behavioral data
* **Generalized canonical correlation results**\
The analyses require that the latent dimensions of rsfMRI time series and individual functional connectivity matrices have been previously computed using the scripts in the [GCCA repository](https://github.com/alberti-f/GCCA).
* **Shaefer 400 parcellation**\
For the graph-theory setion of the analyses the the cortical surface parcellation by [Schaefer and colleagues (2018)](https://people.csail.mit.edu/ythomas/publications/2018LocalGlobal-CerebCor.pdf) which can be found [here](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP)

### Settings file
To make the data available to the scrips their local paths have to be spexified in the *settings.txt.* file,  where other parameters can also be tweaked.

## Requirements
* **Python 3.9**\
The requirements.txt file can be used to install thefollowing dependencies
    * matplotlib
    * nibabel
    * numpy
    * pandas
    * scipy
    * scikit-learn
    * statsmodels
* **Workbench Command**\
Availble [here](https://www.humanconnectome.org/software/get-connectome-workbench)
* **Others**\
The repository also includes:
    * ciftool_FA: a collection of functions used to handle cifti files
    * networkx: the [Networkx 2.8](https://github.com/networkx/networkx/tree/v2.8) package ([Hagberg et al. 2008](http://conference.scipy.org.s3-website-us-east-1.amazonaws.com/proceedings/scipy2008/paper_2/full_text.pdf)) with ad-hoc adaptations