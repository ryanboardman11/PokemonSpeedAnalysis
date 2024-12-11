import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from scipy.stats import ttest_ind

URL = 'https://pokeapi.co/api/v2/pokemon/'
def fetch_pokemon_stats(pokemon_id):
    try:
        response = requests.get(f"{URL}{pokemon_id}/")
        response.raise_for_status()
        pokemon_data = response.json()
        stats = {
            'id': pokemon_id,
            'name': pokemon_data['name'],
            'hp': 0,
            'attack': 0,
            'defense': 0,
            'special-attack': 0,
            'special-defense': 0,
            'speed': 0
        }
        for stat in pokemon_data['stats']:
            stat_name = stat['stat']['name']
            stat_value = stat['base_stat']
            if stat_name in stats:
                stats[stat_name] = stat_value
        return stats
    except requests.HTTPError as e:
        print(f"HTTP Error for ID {pokemon_id}: {e}")
    except requests.RequestException as e:
        print(f"Request Exception for ID {pokemon_id}: {e}")
    except ValueError as e:
        print(f"JSON Decode Error for ID {pokemon_id}: {e}")
    return None

generation_one_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(1, 152) if fetch_pokemon_stats(i) is not None]
generation_two_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(152, 252) if fetch_pokemon_stats(i) is not None]
generation_three_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(252, 387) if fetch_pokemon_stats(i) is not None]
generation_four_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(387, 494) if fetch_pokemon_stats(i) is not None]
generation_five_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(494, 650) if fetch_pokemon_stats(i) is not None]
generation_six_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(650, 722) if fetch_pokemon_stats(i) is not None]
generation_seven_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(722, 810) if fetch_pokemon_stats(i) is not None]
generation_eight_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(810, 906) if fetch_pokemon_stats(i) is not None]
generation_nine_pokemon_stats_list = [fetch_pokemon_stats(i) for i in range(906, 1026) if fetch_pokemon_stats(i) is not None]
df_pokemon_stats_generation_one = pd.DataFrame(generation_one_pokemon_stats_list)
df_pokemon_stats_generation_two = pd.DataFrame(generation_two_pokemon_stats_list)
df_pokemon_stats_generation_three = pd.DataFrame(generation_three_pokemon_stats_list)
df_pokemon_stats_generation_four = pd.DataFrame(generation_four_pokemon_stats_list)
df_pokemon_stats_generation_five = pd.DataFrame(generation_five_pokemon_stats_list)
df_pokemon_stats_generation_six = pd.DataFrame(generation_six_pokemon_stats_list)
df_pokemon_stats_generation_seven = pd.DataFrame(generation_seven_pokemon_stats_list)
df_pokemon_stats_generation_eight = pd.DataFrame(generation_eight_pokemon_stats_list)
df_pokemon_stats_generation_nine = pd.DataFrame(generation_nine_pokemon_stats_list)
df_pokemon_stats_generation_one['generation'] = 1
df_pokemon_stats_generation_two['generation'] = 2
df_pokemon_stats_generation_three['generation'] = 3
df_pokemon_stats_generation_four['generation'] = 4
df_pokemon_stats_generation_five['generation'] = 5
df_pokemon_stats_generation_six['generation'] = 6
df_pokemon_stats_generation_seven['generation'] = 7
df_pokemon_stats_generation_eight['generation'] = 8
df_pokemon_stats_generation_nine['generation'] = 9
combined_pokemon_stats = pd.concat([
    df_pokemon_stats_generation_one,
    df_pokemon_stats_generation_two,
    df_pokemon_stats_generation_three,
    df_pokemon_stats_generation_four,
    df_pokemon_stats_generation_five,
    df_pokemon_stats_generation_six,
    df_pokemon_stats_generation_seven,
    df_pokemon_stats_generation_eight,
    df_pokemon_stats_generation_nine
])

# EDA Data Cleaning (Since we don't need anything besides Name, Attack, Special
# Attack, and Speed we can drop the rest)
combined_pokemon_stats.drop(['hp', 'defense', 'special-defense'], axis=1, inplace=True)
# EDA Check Packaging
# Check records and columns
print("DataFrame shape:", combined_pokemon_stats.shape)
# Check the DataFrame for correct datatypes
print("Data types of each column:")
print(combined_pokemon_stats.dtypes)
# Verify that all columns are present
expected_columns = ['id', 'name', 'attack', 'special-attack', 'speed', 'generation']
if all(column in combined_pokemon_stats.columns for column in expected_columns):
    print("All expected columns are present.")
else:
    missing_columns = set(expected_columns) - set(combined_pokemon_stats.columns)
    print("Missing columns:", missing_columns)
# EDA Check for any missing values
print("Missing values in each column:")
print(combined_pokemon_stats.isnull().sum())
# EDA Summary statistics for numerical columns / Checking the N's to make sure
# There are 1025 in each column.
print("Summary statistics for the DataFrame:")
print(combined_pokemon_stats.describe())
# EDA Validation against external data source. Since I know a lot about Pokémon
# I can tell from looking at the .describe() that everything looks in order.
# The stats are within range for what is normal and all the stats
# are there.
are_all_columns_complete = (combined_pokemon_stats.count() == 1025).all()
print(f"All columns have 1025 entries: {are_all_columns_complete}")
# EDA Check the Top and Bottom
print(combined_pokemon_stats.head())
print(combined_pokemon_stats.tail())


generation_comparison_results = []
# Loop through each generation
for gen in range(1, 10):
    gen_data = combined_pokemon_stats[combined_pokemon_stats['generation'] == gen]

    # Calculate the 90th percentile cutoffs within the generation
    attack_cutoff = gen_data['attack'].quantile(0.90)
    sp_attack_cutoff = gen_data['special-attack'].quantile(0.90)
    high_attack_pokemon = gen_data[gen_data['attack'] >= attack_cutoff]
    high_sp_attack_pokemon = gen_data[gen_data['special-attack'] >= sp_attack_cutoff]
    if not high_attack_pokemon.empty and not high_sp_attack_pokemon.empty:
        t_stat, p_val = ttest_ind(high_attack_pokemon['speed'], high_sp_attack_pokemon['speed'], equal_var=False)
        significant_difference = p_val < 0.05
        result = {
            "Generation": gen,
            "T-statistic": t_stat,
            "P-value": p_val,
            "Significant Difference": "Yes" if significant_difference else "No"
        }
    else:
        result = {
            "Generation": gen,
            "T-statistic": None,
            "P-value": None,
            "Significant Difference": "Insufficient data"
        }
    generation_comparison_results.append(result)

results_df = pd.DataFrame(generation_comparison_results)
print(results_df)

# EDA Visualization

# Bar Graph of Each Generation comparing Average Speed of Physical Attackers vs Special Attackers
combined_pokemon_stats['Type'] = np.where(combined_pokemon_stats['attack'] > combined_pokemon_stats['special-attack'],
                                          'Physical Attacker', 'Special Attacker')
type_order = ['Physical Attacker', 'Special Attacker']
combined_pokemon_stats['Type'] = pd.Categorical(combined_pokemon_stats['Type'], categories=type_order, ordered=True)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, ax in enumerate(axes):
    gen_data = combined_pokemon_stats[combined_pokemon_stats['generation'] == i+1]
    mean_speeds = gen_data.groupby('Type')['speed'].mean().reindex(type_order)
    mean_speeds.plot(kind='bar', ax=ax, color=['blue', 'green'])
    ax.set_title(f'Generation {i+1}')
    ax.set_ylabel('Mean Speed')
    ax.set_xticklabels(mean_speeds.index, rotation=0)
    for index, value in enumerate(mean_speeds):
        ax.text(index, value, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Bar Graph of all generations of Physical Attackers and Special Attackers in Descending Order
aggregated_speeds = pd.DataFrame()

# Calculate mean speeds for each generation and each type, and store them in the DataFrame
for i in range(1, 10):
    gen_data = combined_pokemon_stats[combined_pokemon_stats['generation'] == i]
    mean_speeds = gen_data.groupby('Type')['speed'].mean().reindex(type_order)
    temp_df = pd.DataFrame({
        'Mean Speed': mean_speeds.values,
        'Type': mean_speeds.index,
        'Generation': i
    })
    aggregated_speeds = pd.concat([aggregated_speeds, temp_df])
aggregated_speeds.sort_values(by='Mean Speed', ascending=False, inplace=True)
fig, ax = plt.subplots(figsize=(15, 8))
bars = ax.bar(
    x=np.arange(len(aggregated_speeds)) + 1,
    height=aggregated_speeds['Mean Speed'],
    tick_label=aggregated_speeds.apply(lambda x: f"Gen {x['Generation']} - {x['Type']}", axis=1)
)
bar_colors = ['blue' if 'Physical' in label.get_text() else 'green' for label in ax.get_xticklabels()]
for bar, color in zip(bars, bar_colors):
    bar.set_color(color)
plt.xticks(rotation=90)
ax.set_xlabel('Generation - Type')
ax.set_ylabel('Mean Speed')
ax.set_title('Mean Speed of Physical vs Special Attackers Across Generations')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Scatter plot of all Pokémon speeds
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(combined_pokemon_stats['id'], combined_pokemon_stats['speed'], alpha=0.5, color='purple')
ax.set_xlabel('Pokémon ID')
ax.set_ylabel('Speed')
ax.set_title('Scatter Plot of Pokémon Speed')
plt.show()
