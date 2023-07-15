import gzip
import re
import numpy as np
import pandas as pd

# data from: https://github.com/jakevdp/open-recipe-data/blob/main/recipeitems.json.gz
with gzip.open("../pandas-examples-data/recipeitems.json.gz", "rb") as f:
    json_bytes = f.read()

json_str = json_bytes.decode("utf-8")
json_str[:150]
# '{ "_id" : { "$oid" : "5160756b96cc62079cc2db15" }, "name" : "Drop Biscuits and Sausage Gravy", "ingredients" : "Biscuits\\n3 cups All-purpose Flour\\n2 '

recipes = pd.read_json(json_str, lines=True)
recipes.shape
# (173278, 17)

recipes.head()
#                                     _id                                name                                        ingredients  ... recipeCategory dateModified recipeInstructions
# 0  {'$oid': '5160756b96cc62079cc2db15'}     Drop Biscuits and Sausage Gravy  Biscuits\n3 cups All-purpose Flour\n2 Tablespo...  ...            NaN          NaN                NaN
# 1  {'$oid': '5160756d96cc62079cc2db16'}           Hot Roast Beef Sandwiches  12 whole Dinner Rolls Or Small Sandwich Buns (...  ...            NaN          NaN                NaN
# 2  {'$oid': '5160756f96cc6207a37ff777'}  Morrocan Carrot and Chickpea Salad  Dressing:\n1 tablespoon cumin seeds\n1/3 cup /...  ...            NaN          NaN                NaN
# 3  {'$oid': '5160757096cc62079cc2db17'}               Mixed Berry Shortcake  Biscuits\n3 cups All-purpose Flour\n2 Tablespo...  ...            NaN          NaN                NaN
# 4  {'$oid': '5160757496cc6207a37ff778'}             Pomegranate Yogurt Bowl  For each bowl: \na big dollop of Greek yogurt\...  ...            NaN          NaN                NaN
#
# [5 rows x 17 columns]

recipes.iloc[0]  # first row
# _id                                {'$oid': '5160756b96cc62079cc2db15'}
# name                                    Drop Biscuits and Sausage Gravy
# ingredients           Biscuits\n3 cups All-purpose Flour\n2 Tablespo...
# url                   http://thepioneerwoman.com/cooking/2013/03/dro...
# image                 http://static.thepioneerwoman.com/cooking/file...
# ts                                             {'$date': 1365276011104}
# cookTime                                                          PT30M
# source                                                  thepioneerwoman
# recipeYield                                                          12
# datePublished                                                2013-03-11
# prepTime                                                          PT10M
# description           Late Saturday afternoon, after Marlboro Man ha...
# totalTime                                                           NaN
# creator                                                             NaN
# recipeCategory                                                      NaN
# dateModified                                                        NaN
# recipeInstructions                                                  NaN
# Name: 0, dtype: object

# Collect statistics of characters
recipes.ingredients.str.len().describe()
# count    173278.000000
# mean        244.617926
# std         146.705285
# min           0.000000
# 25%         147.000000
# 50%         221.000000
# 75%         314.000000
# max        9067.000000
# Name: ingredients, dtype: float64

# Which recipe has the longest ingredient list
np.argmax(recipes.ingredients.str.len())
# 135598
recipes.name[np.argmax(recipes.ingredients.str.len())]
# 'Carrot Pineapple Spice &amp; Brownie Layer Cake with Whipped Cream &amp; Cream Cheese Frosting and Marzipan Carrots'

# How many of the recipes are for breakfast foods (using regex)
recipes.description.str.contains('[Bb]reakfast').sum()
# 3524

# How many of the recipes list cinnamon as an ingredient (using regex)
recipes.ingredients.str.contains('[Cc]innamon').sum()
# 10526

# Whether any recipes misspell the ingredient as “cinamon”#
recipes.ingredients.str.contains('[Cc]inamon').sum()
# 11

# Simple recipe recommendation system:
# given a list of ingredients,
# we want to find any recipes that use all those ingredients

spices = [
    'salt',
    'pepper',
    'oregano',
    'sage',
    'parsley',
    'rosemary',
    'tarragon',
    'thyme',
    'paprika',
    'cumin'
]

spice_df = pd.DataFrame(
    {
        spice: recipes.ingredients.str.contains(spice, re.IGNORECASE)
        for spice in spices
    }
)
spice_df.head()
#     salt  pepper  oregano   sage  parsley  rosemary  tarragon  thyme  paprika  cumin
# 0  False   False    False   True    False     False     False  False    False  False
# 1  False   False    False  False    False     False     False  False    False  False
# 2   True    True    False  False    False     False     False  False    False   True
# 3  False   False    False  False    False     False     False  False    False  False
# 4  False   False    False  False    False     False     False  False    False  False

# Find a recipe that uses parsley, paprika, and tarragon (using the query method)

selection = spice_df.query('parsley & paprika & tarragon')
#          salt  pepper  oregano   sage  parsley  rosemary  tarragon  thyme  paprika  cumin
# 2069    False    True    False  False     True     False      True  False     True  False
# 74964   False   False    False  False     True     False      True  False     True  False
# 93768    True    True    False   True     True     False      True  False     True  False
# 113926   True    True    False  False     True     False      True  False     True  False
# 137686   True    True    False  False     True     False      True  False     True  False
# 140530   True    True    False  False     True     False      True   True     True  False
# 158475   True    True    False  False     True     False      True  False     True   True
# 158486   True    True    False  False     True     False      True  False     True  False
# 163175   True    True     True  False     True     False      True  False     True  False
# 165243   True    True    False  False     True     False      True  False     True  False

len(selection)
# 10

# Discover the names of found recipes

recipes.name[selection.index]
# 2069      All cremat with a Little Gem, dandelion and wa...
# 74964                         Lobster with Thermidor butter
# 93768      Burton's Southern Fried Chicken with White Gravy
# 113926                     Mijo's Slow Cooker Shredded Beef
# 137686                     Asparagus Soup with Poached Eggs
# 140530                                 Fried Oyster Po’boys
# 158475                Lamb shank tagine with herb tabbouleh
# 158486                 Southern fried chicken in buttermilk
# 163175            Fried Chicken Sliders with Pickles + Slaw
# 165243                        Bar Tartine Cauliflower Salad
# Name: name, dtype: object
