# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:11:19 2023

@author: anandu
"""

# Function to read the name of the dataset and return two datasets


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def read_data(data):
    # Reading the data
    try:
        df = pd.read_csv(
            r'C:\Users\shanf\OneDrive\Desktop\ads1\%s'%
            data, on_bad_lines='skip')
    except pd.errors.ParserError:
        print(f"Error parsing file {data}")
        return None

    # Transpose the data and name the new column as 'Years'
    t_df = df.transpose().reset_index()
    t_df.columns = t_df.iloc[0]
    t_df = t_df[1:]

    return df, t_df


df, t_df = read_data('climate_change.csv')  # Getting the two datasets
df.head()

# Checking all the available indicators
df['Indicator Name'].unique()
# Transposed DataFrame
t_df.head()

# Creating 3 dataframes with 3 indicators by subsetting the original dataframe
df_ele = df[df['Indicator Name'] ==
            'Electric power consumption (kWh per capita)']
df_agri = df[df['Indicator Name'] ==
             'Agricultural land (% of land area)']
df_popgrow = df[df['Indicator Name'] ==
                'Population growth (annual %)']


# Merging the 3 dataframes
df_merge = pd.concat([df_ele, df_agri, df_popgrow])
df_merge.head()

# Resetting the index
df_merge.reset_index(inplace=True, drop=True)

# replace '..' values with NaN values in data
df_merge.replace('..', np.nan, inplace=True)
# removing the past 4 years since there is very little data
df_merge = df_merge.iloc[:, :-3]

# replacing missing values with 0
df_merge.fillna('0', inplace=True)  # replacing NaN with 0's
# dropping redundant columns
df_merge.drop(['Indicator Code', 'Country Code'], axis=1, inplace=True)

years = df_merge.columns[2:]
# removing wrong values final time
df_merge.replace('..', 0, inplace=True)

# Converting the data to numeric and rounding off to 2 decimals
df_merge[years] = np.abs(df_merge[years].astype('float').round(decimals=2))

# Checking of the size of the 3 datasets
df_merge.groupby(['Indicator Name'])['1990'].size().plot(kind='bar')
plt.title('Size of each group')
plt.xticks(rotation=90)
plt.show()

# Plotting the means of the 3 groups of the data over the years
figure, axes = plt.subplots(1, 2)

df_merge.groupby(['Indicator Name'])['1961'].mean().plot(
    kind='bar', ax=axes[0], color='green')
axes[0].title.set_text('Mean of each group in the year 1961')

df_merge.groupby(['Indicator Name'])['2018'].mean().plot(
    kind='bar', ax=axes[1], color='blue')
axes[1].title.set_text(' Mean of each group in the year 2018')


# Add gap between subplots
plt.subplots_adjust(wspace=0.8)
plt.savefig('change.jpeg')
plt.show()


# Select only numeric columns
numeric_cols = df_merge.select_dtypes(include=np.number).columns.tolist()

# Compute skewness and kurtosis for each numeric column
for col in numeric_cols:
    skewness = skew(df_merge[col])
    kurt = kurtosis(df_merge[col])
    print(f"Column {col}: Skewness = {skewness:.2f}, Kurtosis = {kurt:.2f}")

# diving the data back to 3 datasets to perform summary statistics
df_merge_ele = df_merge[df_merge['Indicator Name']
                        == 'Electric power consumption (kWh per capita)']
df_merge_agri = df_merge[df_merge['Indicator Name'] ==
                         'Agricultural land (% of land area)']
df_merge_popgrow = df_merge[df_merge['Indicator Name']
                            == 'Population growth (annual %)']


# saving the summary statistics in a dataframe
stats = pd.DataFrame()
stats['ele'] = df_merge_ele[years].mean(axis=0).to_frame()
stats['agri'] = df_merge_agri[years].mean(axis=0).to_frame()
stats['popgrow'] = df_merge_popgrow[years].mean(axis=0).to_frame()

# select columns for the years 2000 to 2018
stats = stats.loc['2000':'2018']

# scaling the data since all the groups are not in the same scale
scaler = StandardScaler()

# scaling the data using standard scaler and saving it with full column names
stats_sc = pd.DataFrame(
    scaler.fit_transform(stats),
    columns=[
        'Electric power consumption (kWh per capita)',
        'Agricultural land (% of land area)',
        'Population growth'])

stats_sc.index = stats.index  # set the index to be the same as the stats DataFrame

# plotting the general trend of the 3 groups over the years
plt.figure(figsize=(6, 6))  # size of the image
for i in stats_sc.columns:
    plt.plot(stats_sc.index, stats_sc[i], label=i)  # plotting the 3 lines
plt.legend(loc='best')  # setting the location of the legends
plt.xticks(stats_sc.index, rotation=45)  # changing the x axis values
plt.xlim('2000', '2018')  # set the limits for the x axis
plt.ylabel('Relative Change')  # setting y axis label
plt.title('Change of Indicators over the years all over the world')
plt.savefig('trend.jpeg')  # saving the image
plt.show()

usa = df_merge[df_merge['Country Name'] == 'United States']
usa = usa.drop(['Country Name'], axis=1)
usa = usa.set_index('Indicator Name').T

ind = df_merge[df_merge['Country Name'] == 'India']
ind = ind.drop(['Country Name'], axis=1)
ind = ind.set_index('Indicator Name').T


def plot_country(data, name):
    plt.figure(figsize=(6, 8))
    for i in data.columns:
        # Select only the data for the years 2000 to 2018
        data_range = data.loc['2000':'2018', [i]]

        scaler = StandardScaler()
        # scaling the data using standard scaler and saving it with full column
        # names
        country_sc = pd.DataFrame(
            scaler.fit_transform(data_range), columns=[i])
        country_sc.index = data_range.index
        plt.plot(
            country_sc.index,
            country_sc[i],
            label=i)  # plotting the line
        plt.legend(loc='best')  # setting the location of the legends
        plt.xticks(country_sc.index, rotation=45)  # changing the x axis values
        plt.ylabel('Relative Change')  # setting y axis label
    plt.title(
        'Change of Indicators over the years in %s' %
        name)  # setting the title
    plt.show()


plot_country(usa.loc['2000':'2018'], 'usa')
plot_country(ind.loc['2000':'2018'], 'India')


corr_mat = stats_sc.corr()  # calculating the correlation of the data
fig, ax = plt.subplots(figsize=(10, 10))  # size of the figure
im = ax.imshow(corr_mat, cmap='coolwarm')  # plotting the heatmap

# adding the values inside each cell
for i in range(len(corr_mat)):
    for j in range(len(corr_mat)):
        text = ax.text(j, i, round(
            corr_mat.iloc[i, j], 2), ha="center", va="center", color="black")

# setting the labels
ax.set_xticks(np.arange(len(corr_mat.columns)))
ax.set_yticks(np.arange(len(corr_mat.columns)))
ax.set_xticklabels(corr_mat.columns)
ax.set_yticklabels(corr_mat.columns)

# rotating the x axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# adding the color bar
cbar = ax.figure.colorbar(im, ax=ax)

# setting the title
ax.set_title("Correlation Heatmap")

# saving and showing the image
plt.savefig('correlation_heatmap.png')
plt.show()
