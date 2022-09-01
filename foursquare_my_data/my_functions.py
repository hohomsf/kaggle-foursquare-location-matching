import seaborn as sns
import matplotlib.pyplot as plt
import os
from haversine import haversine

try:
    import recordlinkage as rl
except:
    os.system('pip install recordlinkage')
    import recordlinkage as rl

try:
    import pykakasi
except:
    os.system('pip install pykakasi')
    import pykakasi


def text_preprocess(df):
    """
    Preprocesses necessary text columns before blocking.

    Parameters:
        df (pandas.core.frame.DataFrame): The DataFrame that needs text preprocessing.

    Returns:
        None
    """
    
    # set all strings as lowercases and replace missing values by spaces (on some columns only)
    df.loc[df['name'].isna(), 'name'] = ''
    df['name'] = df['name'].str.lower()
    df['address'] = df['address'].str.lower()
    df['city'] = df['city'].str.lower()
    df.loc[df['city'].isna(), 'city'] = ''
    df.loc[df['country'].isna(), 'country'] = ''
    df['categories'] = df['categories'].str.lower()
    
    # combining name and location to avoid matching too much irrelevant pairs
    df['name_loc'] = df['name'] + ' ' + df['city'] + ' ' + df['country']


def convert_japanese_alphabet(df):
    """
    Transforms Japanese characters in a DataFrame to Romaji (English).

    Parameters:
        df (pandas.core.frame.DataFrame): The DataFrame that needs Japanese character transformation.

    Returns:
        df (pandas.core.frame.DataFrame)
    """
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')  # Convert Hiragana into alphabet
    kakasi.setMode('K', 'a')  # Convert Katakana into alphabet
    kakasi.setMode('J', 'a')  # Convert Kanji into alphabet
    conversion = kakasi.getConverter()

    def convert(row):
        for column in ["name", "address", "city", "state"]:
            try:
                row[column] = conversion.do(row[column])
            except:
                pass
        return row

    # only perform 
    df[df["country"] == "JP"] = df[df["country"] == "JP"].apply(convert, axis=1)
    return df


def create_violinplot(vec):
    """
    Create and display violin plot with specific size.

    Parameters:
        vec (numpy.ndarray): The DataFrame that needs Japanese character transformation.

    Returns:
        None
    """
    sns.set(rc={'figure.figsize':(20,5)})
    sns.violinplot(x=vec)
    plt.show()


def get_score(df, lat_win=3, name_win=3):
    """
    Performs blocking and gets a list of DataFrames containing similarity scores on different columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame from which scores will be generated.
        lat_win (int): Window size on latitude blocking, default 3.
        name_win (int): Window size on name_loc blocking, default 3.

    Returns:
        dfs (list)
    """
    # blocking on 'latitude'
    lat_indexer = rl.Index()
    lat_indexer.sortedneighbourhood(left_on='latitude', window=lat_win)
    lat_pairs = lat_indexer.index(df)

    # blocking on 'name_loc'
    name_indexer = rl.Index()
    name_indexer.sortedneighbourhood(left_on='name_loc', window=name_win)
    name_pairs = name_indexer.index(df)

    # join 2 set of pairs and remove the duplicated ones
    all_pairs = lat_pairs.append(name_pairs)
    all_pairs = all_pairs.drop_duplicates(keep='first')
    
    dfs = []
    if len(all_pairs) > 10000:
        # process data in 30 chunks if there are over 10000 records
        chunks = rl.index_split(all_pairs, 30)
        for chunk in chunks:
            comp = rl.Compare()
            comp.string('name', 'name', label='name_score')
            comp.numeric('latitude', 'latitude', label='latitude_score')
            comp.numeric('longitude', 'longitude', label='longitude_score')
            comp.string('address', 'address', label='address_score')
            comp.string('city', 'city', label='city_score')
            comp.string('country', 'country', label='country_score')
            comp.string('categories', 'categories', label='categories_score')
            chunk_df = comp.compute(chunk, df)
            chunk_df.reset_index(inplace=True)
            dfs.append(chunk_df)
            
    else:
        # if there are only few records, skip chunking the data
        comp = rl.Compare()
        comp.string('name', 'name', label='name_score')
        comp.numeric('latitude', 'latitude', label='latitude_score')
        comp.numeric('longitude', 'longitude', label='longitude_score')
        comp.string('address', 'address', label='address_score')
        comp.string('city', 'city', label='city_score')
        comp.string('country', 'country', label='country_score')
        comp.string('categories', 'categories', label='categories_score')
        score = comp.compute(all_pairs, df)
        score.reset_index(inplace=True)
        dfs.append(score)
    
    return dfs


def get_distance(lat_x, lng_x, lat_y, lng_y):
    """
    Calculates the distance between 2 place entries.

    Parameters:
        lat_x (float): Latitude of the first place entry.
        lng_x (float): Longitude of the first place entry.
        lat_y (float): Latitude of the second place entry.
        lng_y (float): Longitude of the second place entry.

    Returns:
        float
    """
    return haversine((lat_x, lng_x), (lat_y, lng_y))


def get_group(x, match_groups):
    """
    Returns the group that a specific place entry belongs to.

    Parameters:
        x (str): ID of the place entry.
        match_groups (list): Groups of place entries grouped by the entities they are pointing to.

    Returns:
        match_group (set)
    """
    for match_group in match_groups:
        if x in match_group:
            return match_group
