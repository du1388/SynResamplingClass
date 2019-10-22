import os
import sys
import argparse
import pandas as pd
import numpy as np
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve, precision_score, f1_score, recall_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.metrics import geometric_mean_score

def OutputHeader():
    print("===========================================================================================")
    print("Synthetic resampling for machine learning and statical modelling for binary classicifcation")
    print("===========================================================================================")
    print("If you have questions please contact and visit GitHub page: https://github.com/du1388 \n")

def ParseInputs():
    parser = argparse.ArgumentParser(description="Synthetic resampling for machine learning and statical modelling for binary classicifcation.")
    group = parser.add_mutually_exclusive_group()

    # Positional Arguments
    parser.add_argument("path",metavar="path",type=str,nargs=1,help="path to csv file containing features and outcome.")
    parser.add_argument("output_directory",metavar="output_directory",type=str,nargs=1,help="output directory for log and result files.")

    # Optional Arguments
    group.add_argument("-v","--verbose",action="store_true",help="verbose mode, print processing details")
    group.add_argument("-q","--quiet",action="store_true",help="quiet mode, print no warnings and errors")

    parser.add_argument("-t","--target",metavar="<string>",action="store",type=str,nargs=1,help="column header name of target outcome")
    parser.add_argument("-id","--index",metavar="<string>",action="store",type=str,nargs=1,help="column header name of patient/sample index")
    parser.add_argument("-rd","--random_seed",metavar="<int>",action="store", type=int, nargs=1, help="random seed (default is 43)")
    parser.add_argument("-ts","--test_split",metavar="<float>",action="store", type=float, nargs=1, help="testing data percentage (default is 0.25)")
    parser.add_argument("-rm","--resampling_method",metavar="<string>",action="store", type=str, nargs='+', help="list of resampling methods. if not specified all methods are used")
    parser.add_argument("-mm","--modelling_method",metavar="<string>",action="store", type=str, nargs='+', help="list of modelling methods. if not specified all methods are used")
    
    # Initiate
    args = parser.parse_args()
    return args

# Default Config Values
class Config():
    def __init__(self, path, output_path):
        self.path = path
        self.output_directory = output_path
        self.verbose = False
        self.quiet = False
        self.target = "Outcome"
        self.index = "ID"
        self.random_seed = 43
        self.test_split = 0.25
        self.resampling_method = ['RandomOverSampler','ADASYN','SMOTE','BorderlineSMOTE','RandomUnderSampler', 'NearMiss', 'TomekLinks', 'EditedNearestNeighbours','SMOTETomek','SMOTEENN']
        self.modelling_method = ['SVM','LogisticRegression','RandomForest','XGABoost']

def FormatInputs(args):
    if not os.path.isdir(args.output_directory[0]):
        os.mkdir(args.output_directory[0])

    f = open(os.path.join(args.output_directory[0], "logs.txt"), "a+")

    # Get Default
    config = Config(args.path[0],args.output_directory[0])
    config.verbose = args.verbose
    config.quiet = args.quiet

    if args.target:
        config.target = args.target[0]
    else:
        f.write('No target specified, default \"Outcome\" used. \n')
        if not config.quiet:
            if config.verbose:
                print('No target specified, default \"Outcome\" used.')

    if args.index:
        config.index = args.index[0]
    else:
        f.write('No index specified, default \"ID\" used. \n')
        if not config.quiet:
            if config.verbose:
                print('No index specified, default \"ID\" used.')

    if args.random_seed:
        config.random_seed = args.random_seed[0]
    else:
        f.write('No random seed specified, default 43 used. \n')
        if not config.quiet:
            if config.verbose:
                print('No random seed specified, default 43 used.')

    if args.test_split:
        config.test_split = args.test_split[0]
    else:
        f.write('No test split ratio specified, default 0.25 used. \n')
        if not config.quiet:
            if config.verbose:
                print('No test split ratio specified, default 0.25 used.')

    if args.resampling_method:
        if 'All' in args.resampling_method:
            config.resampling_method = config.resampling_method
            f.write("All valid methods used \n")
            if not config.quiet:
                if config.verbose:
                    print("All valid methods used")
        else:
            config.resampling_method = [x for x in args.resampling_method if x in config.resampling_method]
            f.write("Valid specified resampling methods: "+", ".join(config.resampling_method)+"\n")
            if not config.quiet:
                if config.verbose:
                    print("Valid specified resampling methods: "+", ".join(config.resampling_method))
    else:
        f.write("No resampling method variables specified, all valid methods used \n")
        if not config.quiet:
            if config.verbose:
                print("No resampling method variables specified, all valid methods used")

    if args.modelling_method:
        if 'All' in args.modelling_method:
            config.modelling_method = config.modelling_method
            f.write("All valid methods used \n")
            if not config.quiet:
                if config.verbose:
                    print("All valid methods used \n")
        else:
            config.modelling_method = [x for x in args.modelling_method if x in config.modelling_method]
            f.write("Valid specified modelling methods: "+", ".join(config.modelling_method)+"\n")
            if not config.quiet:
                if config.verbose:
                    print("Valid specified modelling methods: "+", ".join(config.modelling_method))
    else:
        f.write("No modelling method variables specified, all valid methods used \n")
        if not config.quiet:
            if config.verbose:
                print("No modelling method variables specified, all valid methods used")
    f.write("\n")
    print()
    f.close()
    return config

def TrainSVM(x_train, y_train, x_train_ori, y_train_ori, x_test, y_test, random_seed):
    
    # Normalise feature
    sc = StandardScaler()
    x_train_ori = sc.fit_transform(x_train_ori)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    # Train Model
    model = SVC(kernel = 'rbf', random_state=random_seed, probability=True, class_weight='balanced', gamma='scale')
    model.fit(x_train, y_train)
    
    # Prediction
    y_prob_train = model.predict_proba(x_train_ori)[:,1]
    y_prob_test = model.predict_proba(x_test)[:,1]
    
    return y_prob_train,y_prob_test

def TrainLogisticRegression(x_train, y_train, x_train_ori, y_train_ori, x_test, y_test, random_seed):
    
    # Normalise feature
    sc = StandardScaler()
    x_train_ori = sc.fit_transform(x_train_ori)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    # Train Model
    model = LogisticRegression(class_weight='balanced', random_state=random_seed, solver='lbfgs')
    model.fit(x_train, y_train)
    
    # Prediction
    y_prob_train = model.predict_proba(x_train_ori)[:,1]
    y_prob_test = model.predict_proba(x_test)[:,1]
    
    return y_prob_train,y_prob_test

def TrainRandomForest(x_train, y_train, x_train_ori, y_train_ori, x_test, y_test, random_seed):
    
    # Normalise feature
    sc = StandardScaler()
    x_train_ori = sc.fit_transform(x_train_ori)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=10, criterion='entropy', class_weight='balanced', random_state=random_seed)
    model.fit(x_train, y_train)
    
    # Prediction
    y_prob_train = model.predict_proba(x_train_ori)[:,1]
    y_prob_test = model.predict_proba(x_test)[:,1]
    
    return y_prob_train,y_prob_test

def TrainXGABoost(x_train, y_train, x_train_ori, y_train_ori, x_test, y_test, random_seed):
    
    # Normalise feature
    sc = StandardScaler()
    x_train_ori = sc.fit_transform(x_train_ori)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    # Train Model
    model = XGBClassifier(nthread=10, max_depth=4, n_estimators=500, learning_rate=0.05, seed=random_seed)
    model.fit(x_train, y_train.reshape(-1))
    
    # Prediction
    y_prob_train = model.predict_proba(x_train_ori)[:,1]
    y_prob_test = model.predict_proba(x_test)[:,1]
    
    return y_prob_train, y_prob_test

def GetOptimalThreshold(y_lab, y_pred):
    fpr,tpr,thres = roc_curve(y_lab, y_pred)
    optimal = thres[np.argmax(tpr+(1-fpr))]
    return optimal

def GetMetrics(y_lab, y_pred, positive=1, optimal=None):
    fpr,tpr,thres = roc_curve(y_lab, y_pred, pos_label=1)    
    
    if optimal:
        pred_lab = np.zeros(len(y_pred))
        pred_lab[y_pred > optimal] = 1
    else:
        pred_lab = np.round(y_pred)
    
    # Metrics
    AUC = auc(fpr,tpr)
    ACC = accuracy_score(y_lab, pred_lab)
    GM = geometric_mean_score(y_lab, pred_lab, pos_label=positive)
    PREC = precision_score(y_lab, pred_lab, pos_label=positive)
    RECL = recall_score(y_lab, pred_lab, pos_label=positive)
    FS = f1_score(y_lab, pred_lab, pos_label=positive)
    
    return AUC, ACC, GM, PREC, RECL, FS

def MatchModelResample(model_name, resample_name=None):
    
    model_case = {
        'SVM': TrainSVM,
        'LogisticRegression': TrainLogisticRegression,
        'RandomForest': TrainRandomForest,
        'XGABoost': TrainXGABoost
    }
    
    resample_case = {
        'No Resampling': None,
        'RandomOverSampler': RandomOverSampler,
        'ADASYN': ADASYN,
        'SMOTE': SMOTE,
        'BorderlineSMOTE': BorderlineSMOTE,
        'RandomUnderSampler': RandomUnderSampler,
        'NearMiss': NearMiss,
        'TomekLinks': TomekLinks,
        'EditedNearestNeighbours': EditedNearestNeighbours,
        'SMOTETomek': SMOTETomek,
        'SMOTEENN' : SMOTEENN
    }
    
    model = model_case[model_name]
    if resample_name:
        resampler = resample_case[resample_name]
    else:
        resampler = None
    
    return model, resampler

class MainClass:
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(config.path, index_col=config.index)
        self.X = self.df.drop(config.target, axis=1)
        self.Y = pd.DataFrame(self.df[config.target])
        
        # Split training/testing data
        x_train, x_test, y_train, y_test = train_test_split(self.X,self.Y,test_size=config.test_split,
                                                            random_state=config.random_seed)
        self.x_train = pd.DataFrame(x_train)
        self.x_test = pd.DataFrame(x_test)
        self.y_train = pd.DataFrame(y_train)
        self.y_test = pd.DataFrame(y_test)
        
        # Export DataFrame to csv
        self.df_train = pd.concat([self.y_train,self.x_train],axis=1,sort=True)
        self.df_train.to_csv(os.path.join(config.output_directory,"training_data.csv"))
        self.df_test = pd.concat([self.y_test,self.x_test],axis=1,sort=True)
        self.df_test.to_csv(os.path.join(config.output_directory,"testing_data.csv"))
        
        self.feature_name = list(self.x_train.columns)
        
        self.x_train_array = self.x_train.to_numpy()
        self.x_test_array = self.x_test.to_numpy()
        self.y_train_array = self.y_train.to_numpy().reshape(-1)
        self.y_test_array = self.y_test.to_numpy().reshape(-1)
        
        # Get class
        self = self.GetCounts()
        
        self.workbook = xlwt.Workbook()
        self = self.RunAnalysis()
        
        self.workbook.save(os.path.join(self.config.output_directory, "results.xls"))
        
    def GetCounts(self):        
        f = open(os.path.join(self.config.output_directory, "logs.txt"), "a+")
        
        # Full Data Count
        f.write("Data Set (n={})\n".format(len(self.Y)))
        if not self.config.quiet:
            if self.config.verbose:
                print("Data Set (n={})".format(len(self.Y)))
        target_group = self.Y.groupby(self.config.target).size()
        target_name = target_group.keys().name
        class_name = list(target_group.keys())
        
        maxind = target_group.idxmax()
        minind = target_group.idxmin()
        self.majority_df = class_name[maxind]
        self.minority_df = class_name[minind]
        
        for ind in range(len(class_name)):
            if ind == maxind:
                class_type = "majority" 
            else:
                class_type = "minority"
                
            f.write("{} {}: {} ({})\n".format(target_name, class_name[ind], target_group[ind],class_type))
            if not self.config.quiet:
                if self.config.verbose:
                    print("{} {}: {} ({})".format(target_name, class_name[ind], target_group[ind],class_type))
        
        # Training Data Count
        f.write("Training Set (n={})\n".format(len(self.y_train)))
        if not self.config.quiet:
            if self.config.verbose:
                print("Training Set (n={})".format(len(self.y_train)))
        target_group = self.y_train.groupby(self.config.target).size()
        target_name = target_group.keys().name
        class_name = list(target_group.keys())
        
        maxind = target_group.idxmax()
        minind = target_group.idxmin()
        self.majority_train = class_name[maxind]
        self.minority_train = class_name[minind]
                    
        for ind in range(len(class_name)):
            if ind == maxind:
                class_type = "majority" 
            else:
                class_type = "minority"
            f.write("{} {}: {} ({})\n".format(target_name, class_name[ind], target_group[ind],class_type))
            if not self.config.quiet:
                if self.config.verbose:
                    print("{} {}: {} ({})".format(target_name, class_name[ind], target_group[ind],class_type))
                    
        # Test Data Count
        f.write("Testing Set (n={})\n".format(len(self.y_test)))
        if not self.config.quiet:
            if self.config.verbose:
                print("Testing Set (n={})".format(len(self.y_test)))
        target_group = self.y_test.groupby(self.config.target).size()
        target_name = target_group.keys().name
        class_name = list(target_group.keys())
        
        maxind = target_group.idxmax()
        minind = target_group.idxmin()
        self.majority_test = class_name[maxind]
        self.minority_test = class_name[minind]
                    
        for ind in range(len(class_name)):
            if ind == maxind:
                class_type = "majority" 
            else:
                class_type = "minority"
            f.write("{} {}: {} ({})\n".format(target_name, class_name[ind], target_group[ind],class_type))
            if not self.config.quiet:
                if self.config.verbose:
                    print("{} {}: {} ({})\n".format(target_name, class_name[ind], target_group[ind],class_type))
                                
        f.close()
        return self
    
    def RunAnalysis(self):
        f = open(os.path.join(self.config.output_directory, "logs.txt"), "a+")
        # Set Style for EXCEL
        borders = xlwt.Borders()
        borders.top = 1
        alignment = xlwt.Alignment()
        alignment.horz = xlwt.Alignment.HORZ_CENTER
        style1 = xlwt.XFStyle()
        style1.borders = borders
        style1.alignment = alignment
        
        borders = xlwt.Borders()
        borders.bottom = 1
        style2 = xlwt.XFStyle()
        style2.borders = borders
        
        resample_list = ["No Resampling"] + self.config.resampling_method
        for ind in range(len(self.config.modelling_method)):
            crt_model_name = self.config.modelling_method[ind]
            ws = self.workbook.add_sheet(crt_model_name)
            ws.write(0,0,label = crt_model_name)
            ws.write(1, 0, label='', style=style1)
            ws.write_merge(1, 1, 1, 9, label = 'Training set', style=style1)
            ws.write_merge(1, 1, 10, 18, label = 'Testing set', style=style1)
            ws.write_merge(2, 2, 4, 6, label = 'Majority class ({} = {})'.format(self.config.target, self.majority_train),style=style2)
            ws.write_merge(2, 2, 7, 9, label = 'Minority class ({} = {})'.format(self.config.target, self.minority_train),style=style2)
            ws.write_merge(2, 2, 13, 15, label = 'Majority class ({} = {})'.format(self.config.target, self.majority_test),style=style2)
            ws.write_merge(2, 2, 16, 18, label = 'Minority class ({} = {})'.format(self.config.target, self.minority_test),style=style2)

            ws.write(3,0, label = 'Resampling methods',style=style2)

            ws.write(3,1, label = 'AUC',style=style2)
            ws.write(3,2, label = 'Accuracy',style=style2)
            ws.write(3,3, label = 'G-Score',style=style2)
            ws.write(3,4, label = 'Precision',style=style2)
            ws.write(3,5, label = 'Recall',style=style2)
            ws.write(3,6, label = 'F-score',style=style2)
            ws.write(3,7, label = 'Precision',style=style2)
            ws.write(3,8, label = 'Recall',style=style2)
            ws.write(3,9, label = 'F-score',style=style2)

            ws.write(3,10, label = 'AUC',style=style2)
            ws.write(3,11, label = 'Accuracy',style=style2)
            ws.write(3,12, label = 'G-Score',style=style2)
            ws.write(3,13, label = 'Precision',style=style2)
            ws.write(3,14, label = 'Recall',style=style2)
            ws.write(3,15, label = 'F-score',style=style2)
            ws.write(3,16, label = 'Precision',style=style2)
            ws.write(3,17, label = 'Recall',style=style2)
            ws.write(3,18, label = 'F-score',style=style2)

            for jnd in range(len(resample_list)):
                crt_resampling_name = resample_list[jnd]

                f.write("Currently running: {} - {}...\n".format(crt_model_name, crt_resampling_name))
                if not self.config.quiet:
                    if self.config.verbose:
                        print("Currently running: {} - {}...".format(crt_model_name, crt_resampling_name))

                model, crt_resampler = MatchModelResample(crt_model_name, crt_resampling_name)

                if crt_resampler:
                    resampler = crt_resampler(random_state=self.config.random_seed)
                    x_train, y_train = resampler.fit_resample(self.x_train_array,self.y_train_array)
                else:
                    x_train, y_train = self.x_train_array,self.y_train_array

                y_train_pred, y_test_pred = model(x_train, y_train, self.x_train_array, self.y_train_array,
                                                 self.x_test, self.y_test, self.config.random_seed)

                thres = GetOptimalThreshold(self.y_train_array, y_train_pred)
                train_performance_majority = GetMetrics(self.y_train_array, y_train_pred, positive=self.majority_train, optimal=thres)
                train_performance_minority = GetMetrics(self.y_train_array, y_train_pred, positive=self.minority_train, optimal=thres)
                test_performance_majority = GetMetrics(self.y_test_array, y_test_pred, positive=self.majority_test, optimal=thres)
                test_performance_minority = GetMetrics(self.y_test_array, y_test_pred, positive=self.minority_test, optimal=thres)
                
                if jnd == len(resample_list)-1:
                    cstyle = style2
                else:
                    cstyle=xlwt.XFStyle()
                
                ctd = jnd + 4
                ws.write(ctd,0, label = crt_resampling_name,style=cstyle)

                ws.write(ctd,1, label = train_performance_majority[0],style=cstyle)
                ws.write(ctd,2, label = train_performance_majority[1],style=cstyle)
                ws.write(ctd,3, label = train_performance_majority[2],style=cstyle)
                ws.write(ctd,4, label = train_performance_majority[3],style=cstyle)
                ws.write(ctd,5, label = train_performance_majority[4],style=cstyle)
                ws.write(ctd,6, label = train_performance_majority[5],style=cstyle)
                ws.write(ctd,7, label = train_performance_minority[3],style=cstyle)
                ws.write(ctd,8, label = train_performance_minority[4],style=cstyle)
                ws.write(ctd,9, label = train_performance_minority[5],style=cstyle)

                ws.write(ctd,10, label = test_performance_majority[0],style=cstyle)
                ws.write(ctd,11, label = test_performance_majority[1],style=cstyle)
                ws.write(ctd,12, label = test_performance_majority[2],style=cstyle)
                ws.write(ctd,13, label = test_performance_majority[3],style=cstyle)
                ws.write(ctd,14, label = test_performance_majority[4],style=cstyle)
                ws.write(ctd,15, label = test_performance_majority[5],style=cstyle)
                ws.write(ctd,16, label = test_performance_minority[3],style=cstyle)
                ws.write(ctd,17, label = test_performance_minority[4],style=cstyle)
                ws.write(ctd,18, label = test_performance_minority[5],style=cstyle)
                
        f.close()

        return self

def Main():
    args = ParseInputs()
    config = FormatInputs(args)

    if not config.quiet:
        OutputHeader()

    program = MainClass(config)
    f = open(os.path.join(program.config.output_directory, "logs.txt"), "a+")
    f.write("Analysis result saved in {}".format(os.path.join(program.config.output_directory,"results.xls")))
    print("Analysis result saved in {}".format(os.path.join(program.config.output_directory,"results.xls")))
    print("Finished.")
if __name__ == '__main__':
    Main()