import copy
import numpy as np
import pandas as pd
import nltk
import re
from   nltk.stem import WordNetLemmatizer
from   scipy.sparse import csr_matrix, hstack
from   sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.svm import SVC
from   sklearn.metrics import accuracy_score
from   sklearn.linear_model import LogisticRegression, SGDClassifier
from   sklearn.naive_bayes import MultinomialNB
from   sklearn import preprocessing
from   sklearn.decomposition import TruncatedSVD
from   sklearn.ensemble import RandomForestClassifier

class DataSet( object ):

    CLASS_MAP = { 
        'false'       : 0,
        'pants-fire'  : 0,
        'barely-true' : 0,
        'half-true'   : 1,
        'mostly-true' : 1,
        'true'        : 1
    }

    ''' Get data from tsvs and perform feature selection '''

    def __init__( self, trainTsv, valTsv, testTsv, colLabels, xLabelCols, yLabelCol, normalizeCols=None ):
        self.colLabels     = colLabels
        self.train         = self.getDfFromTsv( trainTsv )
        self.val           = self.getDfFromTsv( valTsv )
        self.test          = self.getDfFromTsv( testTsv )
        self.xLabelCols    = xLabelCols
        self.yLabelCol     = yLabelCol
        self.normalizeCols = normalizeCols
        self.createDataSet()

    def getDfFromTsv( self, tsv ): 
        ''' convert tsv files to dataframes '''
        header = None if self.colLabels else 0
        return pd.read_csv( tsv, sep='\t', header=header, names=self.colLabels )

    def _normaliseColumnValues( self ):
        if self.normalizeCols:
            normalizer = preprocessing.MinMaxScaler()
            for df in [ self.train, self.val, self.test ]:
                df[ self.normalizeCols ] = normalizer.fit_transform( df[ self.normalizeCols ].fillna( 0.0 ) ) 

    def createDataSet( self ):
        ''' create data set '''
        self._normaliseColumnValues()
        self.train_X = self.train[ self.xLabelCols ].values
        self.train_Y = self.train[ self.yLabelCol ].replace( self.CLASS_MAP ).values
        self.val_X   = self.val[ self.xLabelCols ].values
        self.val_Y   = self.val[ self.yLabelCol ].replace( self.CLASS_MAP ).values
        self.test_X  = self.test[ self.xLabelCols ].values
        self.test_Y  = self.test[ self.yLabelCol ].replace( self.CLASS_MAP ).values

class TextProcessor( object ):

    def __init__( self, dataSet, use_idf=False, ngram_range=(1, 1) ):
        self.dataSet    = dataSet 
        self.vectorizer = TfidfVectorizer( stop_words='english', use_idf=use_idf, ngram_range=ngram_range ) 

    def _applyLemmatization( self, sentences ):
        lemmatizedSentences = []
        for sentence in sentences:
            if not pd.isnull( sentence ):
                cleanedSentence = self._removeSpecialChars( sentence )
                lemmatizer      = WordNetLemmatizer()
                words           = nltk.word_tokenize( cleanedSentence )
                lemmatizedSentences.append( ' '.join( [ lemmatizer.lemmatize( word ) for word in words ] ) )
            else:
                lemmatizedSentences.append( '' )
        return lemmatizedSentences

    def _removeSpecialChars( self, text ):
        return re.sub( r'[^a-zA-Z_\s]+', '', text )

    def vectorize( self, vectorizeCols ):
        ''' vectorize string columns '''
        vectorzisers = []
        for name, arr in [ ( 'train_X', self.dataSet.train_X ), ( 'val_X', self.dataSet.val_X ), ( 'test_X', self.dataSet.test_X ) ]:
            for idx, col in enumerate( vectorizeCols ):
                csr = None
                arr[ :, col ] = self._applyLemmatization( arr[ :, col ] )
                if 'train' in name:
                    vec = copy.deepcopy( self.vectorizer )
                    vec.fit( arr[ :, col ] )   
                    vectorzisers.append( vec )
                csr = hstack( [ csr, vectorzisers[ idx ].transform( arr[ :, col ] ) ] ) if csr else vectorzisers[ idx ].transform( arr[ :, col ] )
                arr = np.delete( arr, col, 1 )
            if arr.shape:
                csr = hstack( [ csr, csr_matrix( arr.astype( float ) ) ] )
            setattr( self.dataSet, name, csr )

class Classifier( object ):

    def __init__( self, classifier ):
        self.classifier = classifier

    def classify( self, dataSet ):
        ''' call classification models '''
        #pca = TruncatedSVD( n_components=500 )
        #pca_comps = pca.fit_transform( dataSet.train_X )
        self.classifier.fit( dataSet.train_X, dataSet.train_Y ) 
        print ( "Accuracy for validation set %s" % ( accuracy_score( dataSet.val_Y, self.classifier.predict( dataSet.val_X ) ) ) )
        print ( "Accuracy for test set %s" % ( accuracy_score( dataSet.test_Y, self.classifier.predict( dataSet.test_X ) ) ) )

def main():
    dirPath   = 'H://Fake News Detection//LIAR-PLUS-master//dataset//'
    trainFile = dirPath + 'train2.tsv'
    valFile   = dirPath + 'val2.tsv'
    testFile  = dirPath + 'test2.tsv'
    dataSet   = DataSet( trainFile, valFile, testFile, colLabels=['Id', 'Label', 'Statement', 'Subject', 'Speaker', 'Job Title', 'State', 'Party', 
                                                                  'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 
                                                                  'Pants on Fire Counts', 'Context', 'Justification'], 

                         xLabelCols=['Statement', 'Subject',  'Speaker', 'Job Title', 'State', 'Party', 'Context', 'Justification',
                                     'Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts'], 
                         yLabelCol='Label',
                         normalizeCols=['Barely True Counts', 'False Counts', 'Half True Counts', 'Mostly True Counts', 'Pants on Fire Counts'] )
    TextProcessor( dataSet, use_idf=True, ngram_range=(1, 1) ).vectorize( [7, 6, 5, 4, 3, 2, 1, 0] ) 
    #classifiers
    print( 'Support Vector Machine' )
    Classifier( SVC( kernel='r', C=0.001 ) ).classify( dataSet )
    print( 'MultinomialNB' )
    Classifier( MultinomialNB() ).classify( dataSet )
    print( 'LogisticRegression' )
    Classifier( LogisticRegression( C=0.1 ) ).classify( dataSet )
    print( 'SGDClassifier' )
    Classifier( SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=100) ).classify( dataSet )
    print( 'RandomForestClassifier' )
    Classifier( RandomForestClassifier(n_estimators=500) ).classify( dataSet )
