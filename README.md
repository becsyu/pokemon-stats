# pokemon-stats
## Summary
This was a data visualisation project featuring seaborn (countplot, histogram, boxplot, chart and scatterplot) on python, with a machine learning training by PCA at the end. Note that this dataset is not intended for machine learning outcome, but rather to practice the technique of reducing dimensions by PCA.
I started with a dataset containing 800 Pokemon game stats and a categorical variable "Type", aiming to visualize the distribution of Pokemon as well as explaining how those of the same "Type" are grouped together. Initial dataframe was treated with a column change - "Type 2" had nearly half the values missing, thus it was replaced with binary 0/1, for data analysis.

Initial data exploration shows that the distribution of types is binomial, which coincides with the distribution of the overall strength of 800 pokemons, measured by the sum of all stats (attack, defense, sp. atk, sp. def and speed). Each stat by itself shows a right-skewed normal distribution, leaving some outliers that are possibly game specials - "legendary pokemons". Having a secondary type present shows a lift in overall strength by 10.8%.

To further examine whether, and if so, which attributes might explain the type, I reduced dimensions using PCA and plotted 18 types on a 2D graph. Although the two principal components are hard to intepret, I was able to cover 74.5% of the variance. I further attempted using PCA to find a fit between features and target, by logistic regression model. The result was of low accurancy - only 20%.

This study shows that game design is sophisticated - when assigning attributes to certain characters, designers must think of whether each attribute fits the distribution, as well as overall strength within each group. Especially with special events, game companies often rewards players with special items and create special levels. These creations are the outliers in game data and must be treated with caution too.
## Resources
For original dataset and more discussion, check out: https://www.kaggle.com/abcsds/pokemon/kernels
## Disclaimer
This was an individual academic project at McCombs Business School at University of Texas, Austin, with no commericial sponsorship. All work is for academic purpose only. Please contact me at sijia.yu@utexas.edu if you have questions.
