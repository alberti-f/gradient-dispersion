#!/bin/bash

echo -e "\ncifti_to_npy" && \
python3 cifti_to_npy.py && \
echo -e "\ninterindividual_dispersion" && \
python3 interindividual_dispersion.py && \
echo -e "\nvariability_clusters" && \
python3 variability_clusters.py && \
echo -e "\ngenerate_csv" && \
python3 generate_csv.py && \
echo -e "\nmodeling_inteligence" && \
python3 modeling_intelligence.py && \
echo -e "\ngraph_analysis" && \
python3 graph_analysis.py && \
echo -e "\nmetric_maps" && \
python3 metric_maps.py && \
echo -e "\n\nDONE"
