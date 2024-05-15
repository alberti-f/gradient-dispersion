#!/bin/bash

echo -e "\n\n\n\ncifti_to_npy\n" > log.txt && \
python3 01.cifti_to_npy.py >> log.txt && \
echo -e "\n\n\n\ninterindividual_dispersion\n" >> log.txt && \
python3 02.interindividual_dispersion.py >> log.txt && \
echo -e "\n\n\n\nvariability_clusters\n" >> log.txt && \
python3 03.variability_clusters.py >> log.txt && \
echo -e "\n\n\n\ngenerate_csv\n"  >> log.txt && \
python3 04.generate_csv.py >> log.txt && \
echo -e "\nmodeling_inteligence\n" >> log.txt && \
python3 05.modeling_intelligence.py >> log.txt && \
echo -e "\n\n\n\ngraph_analysis\n" >> log.txt && \
python3 06.graph_analysis.py >> log.txt && \
echo -e "\n\n\n\nmetric_maps\n" >> log.txt && \
python3 07.metric_maps.py >> log.txt && \
echo -e "\n\n\n\nDONE" >> log.txt 