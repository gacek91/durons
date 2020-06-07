import pandas as pd

def read_log(file, sep="\t", skiprows=3):
    import pandas as pd
    return pd.read_csv(file, sep=sep, skiprows=skiprows)


def fix_names(self):
    'fix column names'
    def strip_accents(text):
        """
        Strip accents from input String.

        :param text: The input string.
        :type text: String.

        :returns: The processed String.
        :rtype: String.
        """
        import unicodedata
        try:
            text = unicode(text, 'utf-8')
        except (TypeError, NameError): # unicode is a default on python 3 
            pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)

    self = self.copy()
    self.columns = self.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.','_')
    c = list(self.columns)
    c = [strip_accents(name) for name in c]
    self.columns = c
    return self

def exclude_nonsubj_data(self, participant):
    'Exclude from the DataFrame any data that does not belong to the subject'
    s = 'subject' in self.columns

        #d = self.copy()
    if s:
        try:
            self = self[self.subject == participant]
        except:
            self = self[self.subject == participant]
    else:
        try:
            self = self[self.Subject == participant]
        except:
            self = self[self.Subject == participant]
    return self
    
def stimuli_from_log(file, participant=None, sep = '\t', skiprows = 3, sort = True):

    '''
    Show all stimuli from the log file
    
    Parameters
    ----------
    
    file : str
        Directory to the log file
    participant : str, default None
        Participant's name - by default reads from the filepath until the first hyphen (-), for example: p55_332-task1.log -> p55_332
    sep : str, default '\t'
        The separator used to read the log file
    skiprows : int, default 3
        Number of rows to skip in the log file
    sort : bool, default True
        Sort the stimuli names
    
    Returns
    -------
    
    List
        A list of stimuli that occured in the log file
        
    
    '''
    import pandas as pd
    import os.path as op
    if participant is None:
        participant = op.split(file)[-1].split("-")[0]


    data = pd.read_csv(file, sep = sep, skiprows = skiprows)
    data.reset_index(inplace=True,drop=True)
    data = fix_names(data)
    data = exclude_nonsubj_data(data, participant = participant)
    return sorted(list(set(data.code))) if sort else list(set(data.code))

def pairwise(iterable):
    from itertools import tee
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))

def pairwise_idx(df, idx):
    from itertools import tee
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    idx = list(idx)
    idx.append(df.index.max())
    a, b = tee(idx)
    next(b, None)
    return list(zip(a, b))

def list2regex(l, pipe = '|'):
    if type(l) is list:
        l = pipe.join(l)
    return l

def get_blocks(file, codes, blocks, until, participant=None, sep = '\t', skiprows = 3, sort = True):
    '''
    Try to get all the stimuli that occur in defined blocks
    
    Parameters
    ----------
    
    file : str
        Directory to the log file
    codes : list or regex
        Names of the stimuli to be searched for within a block
    blocks : list or regex
        Names of the stimuli that mark the begining of a block
    until : list or regex
        Names of the stimuli that mark the end of a block
    participant : str, default None
        Participant's name - by default reads from the filepath until the first hyphen (-), for example: p55_332-task1.log -> p55_332
    sep : str, default '\t'
        The separator used to read the log file
    skiprows : int, default 3
        Number of rows to skip in the log file
    sort : bool, default True
        Sort the stimuli names
    
    Returns
    -------
    
    Dict
        A dictionary with stimuli as keys and block names as values
        
    
    '''
    
    
    import pandas as pd
    import os.path as op
    if participant == None:
        participant = op.split(file)[-1].split("-")[0]
    
    data = pd.read_csv(file, sep = sep, skiprows = skiprows)

    data = fix_names(data)
    data = exclude_nonsubj_data(data, participant = participant)
    data.reset_index(inplace=True,drop=True)
    #data.code = data.code.str.lower()

    codes = list2regex(codes)
    blocks = list2regex(blocks)
    until = list2regex(until)
    
    idx = data[data.code.str.contains(blocks, na=False)].index
    pairs = pairwise_idx(data, idx)
    
    blocks_final = {}
    for pair in pairs:
        d_ = data.iloc[pair[0]:pair[1]]
        task = d_.loc[pair[0]].code
        for c in codes.split('|'):
            if d_.code.str.contains(c).any():
                for s in list(d_.code[d_.code.str.contains(c)]):
                    blocks_final[s] = f"{task}_{c}"
                    
    if sort:
        blocks_final = dict(sorted(blocks_final.items()))
    return blocks_final

def calculate_times(self, s = True):
    d = self
    d.time = d.time.apply(lambda x: float(x))
    d.duration = d.duration.apply(lambda x: float(x))
    if s:
        d.time = (d.time - d.time[d.event_type == 'Pulse'].iloc[0])/10000
        d.duration = d.duration/10000
    else:
        d.time = (d.time - d.time[d.event_type == 'Pulse'].iloc[0])
    d = d[d.time >= 0]
    return d

def durons(file, stimuli=None, participant=None, sep = '\t', skiprows = 3, sort = True, s = True):

    '''
    Calculate durations and onsets in defined blocks
    
    Parameters
    ----------
    
    file : str
        Directory to the log file
    stimuli : dict, default is None
        A dictionary with {stimulus:block_name}. Takes all unique stimuli as separate blocks if no dict provided
    participant : str, default None
        Participant's name - by default reads from the filepath until the first hyphen (-), for example: p55_332-task1.log -> p55_332
    sep : str, default '\t'
        The separator used to read the log file
    skiprows : int, default 3
        Number of rows to skip in the log file
    sort : bool, default True
        Sort the stimuli names
    s : bool, default True
        Calculate the times in seconds (divide the times by 10000)
    
    Returns
    -------
    
    Dict of Pandas DataFrames
        A dictionary with conditions as keys and Pandas DataFrames as values. Each DataFrame contains durations and onsets of the stimuli.
        
    
    '''
    
    import os.path as op
    import pandas as pd

    if participant is None:
        participant = op.split(file)[-1].split("-")[0]

    data = pd.read_csv(file, sep = sep, skiprows = skiprows)
    data = fix_names(data)
    data = exclude_nonsubj_data(data, participant = participant)
    data = calculate_times(data, s = True)
    data = data[['code','time','duration']].sort_values(by=['code', 'time']).reset_index(drop=True)
    data.columns = ['names','onsets','durations']
    if stimuli != None:
        data.names = data.names.map(stimuli)
    data = data.dropna()
    data_temp = data.copy()
    data = {}
    for name in list(set(data_temp.names)):
        data[name] = data_temp[data_temp.names == name][['onsets','durations']].sort_values(by='onsets').reset_index(drop=True)
    if sort:
        data = dict(sorted(data.items()))
    return data

def durons_savemat(durons_dict, filename, sort = True, output = False):
    '''
    Export durations and onsets to a .mat file
    
    Parameters
    ----------
    
    durons_dict : dict
        Dictionary of Pandas DataFrames with durations and onsets
    filename : str
        Filename or path where the mat file will be saved
    sort : bool or list, default True
        Sort the file alphabetically or with a pre-defined order
    output : bool, default False
        Show the final dictionary of arrays to be saved to the .mat file
    '''
    from scipy.io import savemat
    import numpy as np
    mat = {}
    
    if sort:
        names = sorted(durons_dict.keys())
    if not sort:
        names = durons_dict.keys()
    elif type(sort) is list:
        names = sort
    
    mat["names"] = np.array([t for t in names],dtype="object")
    mat["durations"] = np.array([np.array([t for t in durons_dict[condition]['durations']]) for condition in mat['names']])
    mat["onsets"] = np.array([np.array([t for t in durons_dict[condition]['onsets']]) for condition in mat['names']])
    savemat(file_name = filename, mdict = mat)
    if output:
        return mat
    
@pd.api.extensions.register_dataframe_accessor("mr")
class fmri:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
    def fix_names(self):
        self._obj.columns = self._obj.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.','_')

    @staticmethod
    def _validate(obj):
        # verify there is a column Code or code
        if 'Code' not in obj.columns and 'code' not in obj.columns:
            raise AttributeError("Must have 'code' or 'Code' column")

    @property
    def decode(self):
        try:
            code = self._obj.code.str.lower()
        except:
            code = self._obj.Code.str.lower()
        
        for i, c in enumerate(code):
            if 'cue' in c:
                code.iloc[i] = 'cue'
            elif 'block_' in c:
                if 'con' in c:
                    code.iloc[i] = 'control'
                elif 'exp' in c:
                    code.iloc[i] = 'experimental'
                elif 'end' in c:
                    code.iloc[i] = code.iloc[i-1]
        
        return code
    
    def calculate_times(self, s = True):
        d = self._obj
        if s:
            d.time = (d.time - d.time[d.event_type == 'Pulse'].iloc[0])/10000
        else:
            d.time = (d.time - d.time[d.event_type == 'Pulse'].iloc[0])
        return d
    
    def durations(self, s = True):
        durations = []
        d = self._obj
        if s:
            s = 10000
        for row in range(d.shape[0]):
            try:
                if d.iloc[row,:]['code'] == 'cue':
                    durations.append(d.iloc[row,:]['duration']/s)
                elif d.iloc[row, :]['code'] == d.iloc[row + 1, :]['code']:
                    durations.append(d.iloc[row+1,:]['time']-d.iloc[row,:]['time'])
                else:
                    durations.append(np.nan)
            except:
                durations.append(np.nan)
        return durations
    
    def blocknum(self):
        try:
            code = self._obj.code.str.lower()
        except:
            code = self._obj.Code.str.lower()
        n = []
        ni = 1
        for i, c in enumerate(code):
            if i == 0:
                n.append(ni)
                ni += 1
            else:
                if code.iloc[i] != code.iloc[i-1]:
                    if code.iloc[i] != code.iloc[i + 1]:
                        ni += 1
                        n.append(ni)
                        ni += 1
                    else:
                        n.append(ni)
                try:
                    if (code.iloc[i] == code.iloc[i-1]) and (code.iloc[i] != code.iloc[i+1]):
                        n.append(ni)
                except:
                    n.append(ni)
        return n
    
    def get_timepoints(self):
        d = self._obj
        d = d[d.code.str.contains('BLOCK_END')|d.code.str.contains('block_end')|
          d.code.str.contains('BLOCK_START')|d.code.str.contains('block_start')|
          d.code.str.contains('CUE')|d.code.str.contains('cue')]
        return d
    
    def durons(self):
        d = self._obj
        d = d[['code','time','duration']].dropna().sort_values(by=['code', 'time']).reset_index(drop=True)
        d.columns = ['names','onsets','durations']
        d_temp = d.copy()
        d = {}
        for name in list(set(d_temp.names)):
            d[name] = d_temp[d_temp.names == name][['onsets','durations']].reset_index(drop=True)
        return d
    
@pd.api.extensions.register_series_accessor("mr")
class SubclassedSeries(pd.Series):

    @property
    def _constructor(self):
        return SubclassedSeries

    @property
    def _constructor_expanddim(self):
        return SubclassedDataFrame

    
    def blocknum(self):
        code = self.str.lower()
        n = []
        ni = 1
        for i, c in enumerate(code):
            if i == 0:
                n.append(ni)
                ni += 1
            else:
                if code.iloc[i] != code.iloc[i-1]:
                    if code.iloc[i] != code.iloc[i + 1]:
                        ni += 1
                        n.append(ni)
                        ni += 1
                    else:
                        n.append(ni)
                try:
                    if (code.iloc[i] == code.iloc[i-1]) and (code.iloc[i] != code.iloc[i+1]):
                        n.append(ni)
                except:
                    n.append(ni)
        return n
    
    def decode(self):
        code = self.str.lower()
        for i, c in enumerate(code):
            if 'cue' in c:
                code.iloc[i] = 'cue'
            elif 'block_' in c:
                if 'con' in c:
                    code.iloc[i] = 'control'
                elif 'exp' in c:
                    code.iloc[i] = 'experimental'
                elif 'end' in c:
                    code.iloc[i] = code.iloc[i-1]
        return code