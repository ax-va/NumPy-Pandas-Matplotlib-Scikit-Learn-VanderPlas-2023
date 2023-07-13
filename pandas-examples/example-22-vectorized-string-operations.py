import numpy as np
import pandas as pd

# motivation: need of vectorized string operations

x = np.array([2, 3, 5, 7, 11, 13])
x * 2
# array([ 4,  6, 10, 14, 22, 26])

data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]
# ['Peter', 'Paul', 'Mary', 'Guido']

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
# [s.capitalize() for s in data]
# AttributeError: 'NoneType' object has no attribute 'capitalize'
[s if s is None else s.capitalize() for s in data]
# ['Peter', 'Paul', None, 'Mary', 'Guido']
# verbose, inconvenient, and error-prone

names = pd.Series(data)
names.str.capitalize()
# 0    Peter
# 1     Paul
# 2     None
# 3     Mary
# 4    Guido
# dtype: object

monte = pd.Series([
    'Graham Chapman',
    'John Cleese',
    'Terry Gilliam',
    'Eric Idle',
    'Terry Jones',
    'Michael Palin',
])

# methods similar to Python string methods

# The following Pandas str methods mirror Python string methods:
# len           lower           translate       islower         ljust
# upper         startswith      isupper         rjustfind       findm
# isdecimal     zfill           index           isalpha         split
# strip         rindex          isdigit         rsplit          rstrip
# capitalize    isspace         partition       lstrip          swapcase

monte.str.lower()
# 0    graham chapman
# 1       john cleese
# 2     terry gilliam
# 3         eric idle
# 4       terry jones
# 5     michael palin
# dtype: object

monte.str.len()
# 0    14
# 1    11
# 2    13
# 3     9
# 4    11
# 5    13
# dtype: int64

monte.str.startswith('T')
# 0    False
# 1    False
# 2     True
# 3    False
# 4     True
# 5    False
# dtype: bool

monte.str.split()
# 0    [Graham, Chapman]
# 1       [John, Cleese]
# 2     [Terry, Gilliam]
# 3         [Eric, Idle]
# 4       [Terry, Jones]
# 5     [Michael, Palin]
# dtype: object

# methods using regular expressions

# match     calls re.match on each element, returning a Boolean
# extract   calls re.match on each element, returning matched groups as strings
# findall   calls re.findall on each element
# replace   replaces occurrences of pattern with some other string
# contains  calls re.search on each element, returning a boolean
# count     counts occurrences of pattern
# split     is equivalent to str.split, but accepts regexps
# rsplit    is equivalent to str.rsplit, but accepts regexps

monte.str.extract('([A-Za-z]+)', expand=False)
# 0     Graham
# 1       John
# 2      Terry
# 3       Eric
# 4      Terry
# 5    Michael
# dtype: object

monte.str.extract('([A-Za-z]+)', expand=True)
#          0
# 0   Graham
# 1     John
# 2    Terry
# 3     Eric
# 4    Terry
# 5  Michael

# Find all names that start and end with a consonant, making use
# of the start-of-string (^) and end-of-string ($) regex characters
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
# 0    [Graham Chapman]
# 1                  []
# 2     [Terry Gilliam]
# 3                  []
# 4       [Terry Jones]
# 5     [Michael Palin]
# dtype: object

# additional methods

# get               indexes each element
# slice             slices each element
# slice_replace     replaces slice in each element with the passed value
# cat               concatenates strings
# repeat            repeats values
# normalize         returns Unicode form of strings
# pad               adds whitespace to left, right, or both sides of strings
# wrap              splits long strings into lines with length less than a given width
# join              joins strings in each element of the Series with the passed separator
# get_dummies       extracts dummy variables as a DataFrame

# df.str.slice(0, 3) is equivalent to df.str[0:3]
monte.str[0:3]
# 0    Gra
# 1    Joh
# 2    Ter
# 3    Eri
# 4    Ter
# 5    Mic
# dtype: object

# df.str.get(i) s equivalent to df.str[i]
monte.str[0]
# 0    G
# 1    J
# 2    T
# 3    E
# 4    T
# 5    M
# dtype: object

monte.str.split().str[-1]
# 0    Chapman
# 1     Cleese
# 2    Gilliam
# 3       Idle
# 4      Jones
# 5      Palin
# dtype: object

# indicator variables: get_dummies

indicator_codes = ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']

full_monte = pd.DataFrame(
    {
        'name': monte,
        'info': indicator_codes
    }
)
#              name   info
# 0  Graham Chapman  B|C|D
# 1     John Cleese    B|D
# 2   Terry Gilliam    A|C
# 3       Eric Idle    B|D
# 4     Terry Jones    B|C
# 5   Michael Palin  B|C|D

# Get a DataFrame with indicator codea and variables

full_monte['info'].str.get_dummies('|')
#    A  B  C  D
# 0  0  1  1  1
# 1  0  1  0  1
# 2  1  0  1  0
# 3  0  1  0  1
# 4  0  1  1  0
# 5  0  1  1  1
