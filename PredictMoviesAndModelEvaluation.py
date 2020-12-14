from SplitDatasetAndBuiltAntiTestsetForUser import SplitDatasetAndBuiltAntiTestsetForUser
from surprise import accuracy


class PredictMoviesAndModelEvaluation:
    MLModels = []

    def __init__(self, data, rnk):
        num = SplitDatasetAndBuiltAntiTestsetForUser(data, rnk)
        self.data = num

    def recommendationSampleTopNData(self, model, fetchModel, UserIDToRecommend):

        print("\nSingular Value Decomposition from Surpise Library...  ")

        train = self.data.getCompleteTrainDataset()
        model.fit(train)
        # In test set we are passing Anti test set for User
        test = self.data.buildAntiTestsetForUser(UserIDToRecommend)
        # predict movies for testset
        predictedMovies = model.test(test)

        recommendedMovies = []

        for userId, mvId, actualRating, calcuatedRating, _ in predictedMovies:
            mvId = int(mvId)
            recommendedMovies.append((mvId, calcuatedRating))

        recommendedMovies.sort(key=lambda x: x[1], reverse=True)
        print("\nOur Recommendation model will recommend below top 5 movies for UserID:", userId)
        for col in recommendedMovies[:5]:
            print(fetchModel.getNameOfMovie(col[0]), "\n", round(col[1], 4), "\n")

    def evaluateModel(self, model):

        recommendationMetrics = {}
        print("\nEvaluating accuracy of SVD model based on MAE, MSE and RMSE score...")
        print("\nIf we have Mean Absolute Error, Mean Square Error and Root Mean Square "
              "lower score then model is good in prediction...")
        model.fit(self.data.getTrainingData())
        predictions = model.test(self.data.getTestData())
        # calculate WAE score suing MAE function from the surprise library
        recommendationMetrics["Mean Absolute Error"] = accuracy.mae(predictions)
        recommendationMetrics["Mean Square Error"] = accuracy.mse(predictions)
        recommendationMetrics["Root Mean Square Error"] = accuracy.rmse(predictions)

