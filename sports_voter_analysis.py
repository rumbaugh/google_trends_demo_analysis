import numpy as np
import pandas as pd


class Demo:
    def __init__(self, masterfile='NFL_fandom_data-google_trends.csv', basepath='./', region_dict=None):
        self.df = pd.read_csv(basepath + masterfile , skiprows=1)
        if region_dict == None:
            region_dict = {'South': np.array(['AL','AR','FL','GA','KY','LA','MS','NC','SC','OK','TN','TX','VA']), 'West': np.array(['AZ','CA','CO','ID','MT','NV','NM','OR','UT','WA','WY']), 'East': np.array(['CT','DE','DC','ME','MD','MA','NH','NJ','NY','PA','RI','VT']), 'Midwest': np.array(['IL','IN','IA','KS','MI','MO','NE','ND','SD','OH','WI','MN'])} 
        self.region_dict = region_dict
        self.state_to_region_dict = {}
        for reg, states in region_dict.items():
            for state in states: self.state_to_region_dict[state] = reg
        self.set_states()
        self.set_regions()
        
    def parse_state(self, instring):
        strarr = np.array(list(instring))
        upperarr, lowerarr = np.array(list(instring.upper())), np.array(list(instring.lower()))
        capitals = (strarr == upperarr) & (strarr != lowerarr)
        state_starts = capitals[:-1] & capitals [1:]
        num_states = np.sum(state_starts)
        if num_states == 0:
            print 'No states found in ' + instring
            return
        states = np.zeros(num_states, dtype='|S2')
        for s, state_start in zip(np.arange(0, num_states), np.arange(0, len(instring)-1)[state_starts]):
            states[s] = instring[state_start:state_start+2]
        return states

    def set_states(self):
        locations = self.df.DMA.values
        self.df['state'] = np.zeros(len(df), dtype='|S8')
        for i in range(0, len(df)):
            self.df.state.iloc[i] = ','.join(self.parse_state(locations[i]))

    def set_regions(self):
        try:
            self.df.state
        except AttributeError:
            print 'Must set states'
            return
        self.df['region'] = np.zeros(len(df), dtype='|S24')
        for i in range(0, len(df)): 
            cur_states = self.df.state.iloc[i].split(',')
            try:
                self.df.region.iloc[i] = self.state_to_region_dict[cur_states[0]]
            except KeyError:
                self.df.region.iloc[i] = 'Other'
            for i_state in range(1, len(cur_states)):
                try:
                    curreg = self.state_to_region_dict[cur_states[i_state]]
                except KeyError:
                    curreg = 'Other'
                if not(curreg in self.df.region.iloc[i].split(',')):
                    self.df.region.iloc[i] = '{},{}'.format(self.df.region.iloc[i],curreg)


                
                



    
        
