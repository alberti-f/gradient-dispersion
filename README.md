# Understanding the link between functional profiles and intelligence through dimensionality reduction and graph analysis
Francesco Alberti, Arianna Menardi, Daniel S. Margulies, Antonino Vallesi


This repository contains the scripts used to perform the analyses of a study investigating the relationdship between interindividual variability of intelligence and the functional organization of the cerebral cortex.\
[Read the preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.12.536421v1)
## Abstract

> _There is a growing interest in neuroscience for how individual-specific structural and functional features of the cortex relate to cognitive traits. This work builds on previous research which, by using classical high-dimensional approaches, has proven that the interindividual variability of functional connectivity (FC) profiles reflects differences in fluid intelligence. To provide an additional perspective into this relationship, the present study uses a recent framework for investigating cortical organization: functional gradients. This approach places local connectivity profiles within a common low-dimensional space whose axes are functionally interpretable dimensions. Specifically, this study uses a data-driven approach to model the association between FC variability and interindividual differences in intelligence. For one of these loci, in the right ventral-lateral prefrontal cortex (vlPFC), we describe an association between fluid intelligence and the relative functional distance of this area from sensory and high-cognition systems. Furthermore, the topological properties of this region indicate that, with decreasing functional affinity with high-cognition systems, vlPFC functional connections are more evenly distributed across all networks. Participating in multiple functional networks may reflect a better ability to coordinate sensory and high-order cognitive systems._

![alt text](https://github.com/alberti-f/gradient-dispersion/blob/main/methods_figure.png)

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


## Settings file
To make the data available to the scrips their local paths have to be spexified in the *settings.txt.* file,  where other parameters can also be tweaked.
