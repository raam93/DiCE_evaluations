# DiCE_evaluations

## Steps to generate Figure 1 and 2 in the paper:

1. Run **run_DiCE_experiments_figure1and2.py:** This will generate CFs for given data and configuration and stores them in **figure1_experiment_results** folder. It also runs Figure 2 experiments and stores the results in **figure2_experiment_results** folder. The code needs to be edited slightly (should add data characteristics and model path) if we need to test with data other than the four we used. For our sample data, I have **already run this code and have stored the results.** 

2. Run **get_figure1_summary_stats.py:** This will compute the summary statistics required for plotting Figure 1 and store them in **figure1_summary_stats** folder.

3. Run **run_LIME_experiments:** This will generate LIME explanations and store them in **lime_explanations** folder.

4. Run **get_figure2_summary_stats.py:** This will compute the summary statistics required for plotting Figure 2 and store them in **figure2_summary_stats** folder. The code needs to be edited slightly if we need to plot Figure 2 for linear models.

5. Run **DiCE_evaluation_plotting.ipynb:** This notebook generates Figure 1 and 2 plots. 

## Folder description:

1. **datasets:** Contains COMPAS, Lending Club, and German credit card data. The code downloads Adult income data from the internet.
2. **paper_plots:**: Contains the plots that are in the paper.
3. **stored_ml_models:** Contains the trained ML models of the four datasets.

Please ignore **get_mixedIntCF_summary_stats.py** and **mixedIntCF_results** as they are incomplete. These are used in generating Figure 1 plots for linear models.

