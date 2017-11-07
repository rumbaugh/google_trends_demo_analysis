import numpy as np
import pandas as pd

from sklearn import decomposition,neighbors
from sklearn.neighbors import KDTree,BallTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from plotting import *

class Demo:
    def __init__(self, masterfile='NFL_fandom_data-google_trends.csv', basepath='./', region_dict=None):
        self.df = pd.read_csv(basepath + masterfile , skiprows=1)
        if region_dict == None:
            region_dict = {'South': np.array(['AL','AR','FL','GA','KY','LA','MS','NC','SC','OK','TN','TX','VA']), 'West': np.array(['AZ','CA','CO','ID','MT','NV','NM','OR','UT','WA','WY']), 'East': np.array(['CT','DE','DC','ME','MD','MA','NH','NJ','NY','PA','RI','VT','WV']), 'Midwest': np.array(['IL','IN','IA','KS','MI','MO','NE','ND','SD','OH','WI','MN'])} 
        self.region_dict = region_dict
        self.state_to_region_dict = {}
        for reg, states in region_dict.items():
            for state in states: self.state_to_region_dict[state] = reg
        self.set_states()
        self.set_regions()
        self.NCV=5
        
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
        self.df['state'] = np.zeros(len(self.df), dtype='|S8')
        for i in range(0, len(self.df)):
            self.df.state.iloc[i] = ','.join(self.parse_state(locations[i]))

    def set_regions(self):
        try:
            self.df.state
        except AttributeError:
            print 'Must set states'
            return
        self.df['region'] = np.zeros(len(self.df), dtype='|S24')
        for i in range(0, len(self.df)): 
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

    def reg_classify(self,technique = 'RF', usevotes = False, remove_regs_other = True, remove_mult_regs = True, sports = ['NFL', 'NBA', 'MLB', 'NHL', 'NASCAR', 'CBB', 'CFB'],train_perc=0.9, NE=10, NN=5, algorithm = 'kd_tree'):
        #Attempts to classify regions based on the percentage
        #of web traffic for different sports leagues, plus
        #optionally Trump 2016 Vote % as well
        if ((usevotes) & (not('Trump 2016 Vote' in sports))): sports = np.append(sports, 'Trump 2016 Vote')
        X = self.df[sports]
        regions = self.df.region
        if remove_regs_other:
            X, regions = X[regions != 'Other'], regions[regions != 'Other']
        regions_list = regions.unique()
        if remove_mult_regs:
            for reg in regions_list:
                if ',' in reg:
                    X, regions, regions_list = X[regions != reg], regions[regions != reg], regions_list[regions_list != reg]
        if technique == 'RF':
            self.RFClassify(X,regions,train_perc,NE)
        elif technique == 'NN':
            self.NNClassify(X,regions,train_perc,NN,algorithm)
        else:
            print "Invalid technique chosen. Current options are: RF, NN."
            return

    
    def RFClassify(self,X,target,train_perc=0.9,NE=10):
        #Perform random forest classification 
        #Uses a fraction of the sports array as the training set, defined
        #by train_perc. The rest is used as the test set
        train_df = X[:int(train_perc*np.shape(target)[0])]
        train_X,train_y=train_df, target[:int(train_perc*len(target))]
        test_X,test_y=X[int(train_perc*np.shape(X)[0]):],target[int(train_perc*np.shape(X)[0]):]
        self.clf=RandomForestClassifier(n_estimators=NE)
        self.clf.fit(train_X,train_y)
        self.predicted_regions=self.clf.predict(test_X)
        self.CV_scores=cross_val_score(self.clf,X,target,cv=self.NCV)
    
    def NNClassify(self,X,target,train_perc=0.9,n_neighbors=5, algorithm='kd_tree',weights='distance'):
        #Perform nearest neighbor classification
        #Uses a fraction of the sports array as the training set, defined
        #by train_perc. The rest is used as the test set
        train_df = X[:int(train_perc*np.shape(target)[0])]
        train_X,train_y=train_df, target[:int(train_perc*len(target))]
        test_X,test_y=X[int(train_perc*np.shape(X)[0]):],target[int(train_perc*np.shape(X)[0]):]
        self.clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights,algorithm=algorithm)
        self.clf.fit(train_X,train_y)
        self.predicted_y=self.clf.predict(test_X)
        self.CV_scores=cross_val_score(self.clf,X,target,cv=self.NCV)

    def region_vote_estimate(self):
        self.region_vote_dict = {reg: np.median(self.df['Trump 2016 Vote'][self.df.region == reg]) for reg in self.region_dict.keys()}
        
    def BayesP(self, n_obs, observed_dict, prior_dict, search_grid, prob_dens = None):
        try:
            self.region_vote_dict
        except AttributeError:
            print "Set region_vote_dict"
            return
        if len(prior_dict.keys()) != len(observed_dict.keys()):
            print "observed_dict and prior_dict don't have same number of keys"
            return
        if np.count_nonzero(np.sort(prior_dict.keys()) != np.sort(observed_dict.keys())) > 0:
            print "observed_dict and prior_dict don't have same keys"
            return
        if np.shape(search_grid)[1] != len(observed_dict.keys()):
            print 'search grid dimensions invalid'
            return
        n_comp, n_srch = len(prior_dict.keys()), np.shape(search_grid)[0]
        if np.shape(prob_dens) == ():
            if n_comp > 2:
                print "Must specify prob_dens if n_comp > 2"
                return
            search_grid = search_grid[np.argsort(search_grid[:,0])]
            prob_dens = np.zeros(n_srch)
            prob_dens[0], prob_dens[-1] = 0.5 * search_grid[0][0] + 0.5 * search_grid[1][0], 1 - 0.5 * search_grid[-1][0] - 0.5 * search_grid[-2][0] 
            prob_dens[1:-1] = 0.5 * (search_grid[2:,0] - search_grid[:-2])
        outprob = np.zeros(np.shape(search_grid)[1])
