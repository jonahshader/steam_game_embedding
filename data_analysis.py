import csv
import matplotlib.pyplot as plt


def load_steam_app_data(filename):
    """Load steam_app_data.csv into a pandas DataFrame."""
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    return data


def compute_duplicate_count(data, attribute1, attribute2):
    equal = 0
    total = 0
    for row in data:
        if row[attribute1] == row[attribute2]:
            equal += 1
        total += 1
    return equal / total


if __name__ == '__main__':
    data = load_steam_app_data('data/download/steam_app_data.csv')
    for r in data:
        if 'Counter-Strike' in r['name']:
            print(r['name'])
            print(r['detailed_description'])
            print(r['about_the_game'])
            print(r['short_description'])

    print(compute_duplicate_count(data, 'detailed_description', 'about_the_game'))
    print(compute_duplicate_count(data, 'detailed_description', 'short_description'))
    print(compute_duplicate_count(data, 'about_the_game', 'short_description'))

    detailed_description_lengths = []
    about_the_game_lengths = []
    short_description_lengths = []
    for r in data:
        detailed_description_lengths.append(len(r['detailed_description']))
        about_the_game_lengths.append(len(r['about_the_game']))
        short_description_lengths.append(len(r['short_description']))

    print('top 10 longest detailed_description_lengths')
    data_sorted = data.copy()
    data_sorted.sort(key=lambda x: len(x['detailed_description']))
    # data_sorted.reverse()
    for r in data_sorted[:10]:
        print(r['name'], len(r['detailed_description']))

    print('top 10 longest about_the_game_lengths')
    data_sorted = data.copy()
    data_sorted.sort(key=lambda x: len(x['about_the_game']))
    data_sorted.reverse()
    for r in data_sorted[:10]:
        print(r['name'], len(r['about_the_game']))

    # create three histograms of the lengths of the three descriptions
    plt.hist(detailed_description_lengths, bins=100)
    plt.title('Detailed Description Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(about_the_game_lengths, bins=100)
    plt.title('About the Game Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(short_description_lengths, bins=100)
    plt.title('Short Description Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()
