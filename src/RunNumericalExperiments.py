# Run the numerical experiments outlined in Bjarkason (2018)
# Coded by: Elvar K. Bjarkason (2018)

import os

# Create low-rank plus noise test matrices
os.system('python CreateLowRankNoisyTestMatrices.py')
# Plot Figure 1:
os.system('python plotFigure1.py')

# Run 1-view experiments:
os.system('python run1viewExperiments.py')
# Plot some results, i.e. Figures 7-10:
os.system('python plot1viewFigure7.py')
os.system('python plot1viewFigure8.py')
os.system('python plot1viewFigures9and10.py')

# Run Gen. Subspace Iteration Experiments:
os.system('python runGenSubAlg2standardExperiments.py')
# Plot Figure 2:
os.system('python plotFigure2.py')

# Run Gen. Subspace Iteration Experiments on Normal Matrix:
os.system('python runGenSubAlg2OnNormalExperiments.py')
# Plot Figures 3 and 4:
os.system('python plotFigure3.py')
os.system('python plotFigure4.py')

# Run Gen. Subspace Iteration Experiments on Normal Matrix:
os.system('python runBlockKrylovExperiments.py')
# Plot Figures 5 and 6:
os.system('python plotFigures5and6.py')

