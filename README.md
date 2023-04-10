# Cycling Equity
Code repository for senior thesis project

*Equity, Mobility, and Sustainability: Analyzing Geographic and Demographic Disparities in Urban Bikeability*

## Data
Fine-grained data was pulled from OpenStreetMap and saved using the `V2_save_raw_graph.ipynb` notebook. 

`scripts/V2_coarse_graph_parallel.py` converts fine-grained street data for a city into a coarse-grained network at the census tract level. 

Note that running the script requires access to data saved with `V2_save_raw_graph.ipynb`, as well as multiprocessing capabilities. `coarsify.slurm` is a sample slurm script that runs `V2_coarse_graph_parallel.py` on a computer cluster.

Pre-calculated coarse-grained graphs for San Francisco, Philadelphia, and Detroit are saved using `pickle` in `/data/`

## Analysis
`network_info.ipynb` plots coarse-grained street networks and prints information about the size of fine-grained vs. coarse-grained networks. 
See Table 4.1 and Figures 4.1-4.3 in paper.

`bikeability_general.ipynb` creates demonstrative plots of Pareto fronts, network-wide bikeability curves, and "elbow" points on bikeability curves. 
Also calculates bikeability curves and scores for the three case study cities. See Table 5.1 and Figures 4.4-4.6, 5.1 in paper.

`equity.ipynb` performs analysis of geographic equity and demographic equity (racial/socioeconomic). See Tables 5.2, 5.3 and Figures 5.2-5.6 in paper.

`accessibility.ipynb` performs bikeability analysis with an alternate discomfort function that accounts for accessibility needs. See Figures 5.7, 5.8 in paper.

`bikeability_functions.py` common functions used in bikeability calculations

`dem_functions.py` common functions used in demographic analysis
