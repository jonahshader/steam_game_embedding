from data_analysis import load_steam_app_data

data = load_steam_app_data('data/download/steam_app_data.csv')

def game_name_prompt(name=None):
    """Search for a game by name and display games with names that contain the search term.
    If multiple games are found, prompt the user to select one of the games.
    Return game and original index of game in data."""
    if name is None:
        search_term = input('Enter a game name to search for: ')
    else:
        search_term = name
    search_term = search_term.lower()
    search_results = []
    for game in data:
        if search_term in game['name'].lower():
            search_results.append(game)
    if len(search_results) == 0:
        print('No games found')
    elif len(search_results) == 1 or search_term in [game['name'].lower() for game in search_results]:
        # determine original index of game in data
        for i, game in enumerate(data):
            if game['name'] == search_results[0]['name']:
                return game, i
    else:
        print('Multiple games found:')
        for i, game in enumerate(search_results):
            print(f'{i}: {game["name"]}')
        game_index = int(input('Select a game: '))

        # determine original index of game in data
        for i, game in enumerate(data):
            if game['name'] == search_results[game_index]['name']:
                return game, i