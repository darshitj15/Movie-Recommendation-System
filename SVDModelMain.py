from MovieDataSetFeatures import MovieDataSetFeatures
from surprise import SVD
from PredictMoviesAndModelEvaluation import PredictMoviesAndModelEvaluation


def readMovieDataSet():

    movieFeatures = MovieDataSetFeatures()
    print("Data Loading for model >>>")
    modelData = movieFeatures.readDataset()
    movieRanks = movieFeatures.getRanksBasedOnPopularity()
    return movieFeatures, modelData, movieRanks


movieFeatures, modelData, movieRanks = readMovieDataSet()
modelEval = PredictMoviesAndModelEvaluation(modelData, movieRanks)

SVDModel = SVD()

recommend_user = int(input("Please enter the user ID: "))
modelEval.recommendationSampleTopNData(SVDModel, movieFeatures, recommend_user)

modelEval.evaluateModel(SVDModel)
