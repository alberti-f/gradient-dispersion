#!/bin/bash

echo -e "\n\n\n\ncifti_to_npy\n" && \
python3 01.cifti_to_npy.py && \
echo -e "\n\n\n\ninterindividual_dispersion\n" && \
python3 02.interindividual_dispersion.py && \
echo -e "\n\n\n\nvariability_clusters\n" && \
python3 03.variability_clusters.py && \
echo -e "\n\n\n\ngenerate_csv\n"  && \
python3 04.generate_csv.py && \
echo -e "\nmodeling_inteligence\n" && \
python3 05.modeling_intelligence.py && \
echo -e "\n\n\n\ngraph_analysis\n" && \
python3 06.graph_analysis.py && \
echo -e "\n\n\n\nmetric_maps\n" && \
python3 07.metric_maps.py && \
echo -e "\n\n\n\nDONE" > log.txt