from surprise.model_selection import train_test_split


class SplitDatasetAndBuiltAntiTestsetForUser:

    def __init__(self, records, ratingsBasedOnPopularMovies):
        self.rankings = ratingsBasedOnPopularMovies

        # Do not split the dataset into folds and just return a trainset as is, built from the whole dataset.
        self.trainingDataset = records.build_full_trainset()

        # split dataset into 80/20 format
        self.train, self.test = train_test_split(records, test_size=.20, random_state=10, shuffle=True)

    def getTrainingData(self):
        return self.train

    def getTestData(self):
        return self.test

    def getCompleteTrainDataset(self):
        return self.trainingDataset

    # estimated rating for all the user-item rating pairs that was missing from the dataset.
    def buildAntiTestsetForUser(self, addUserID):
        train = self.trainingDataset
        fillTheMissingvalues = []

        # take global mean of train dataset ratings using surprise library functions
        mean = train.global_mean

        # Convert a user raw id to an inner id
        user = train.to_inner_uid(str(addUserID))

        #ur is a user ratings from the Trainset Surprise library
        user_recommend = set([val for (val, _) in train.ur[user]])

        # check if the movie is already present in the train dataset and if not then return that movie.
        fillTheMissingvalues = fillTheMissingvalues + [(train.to_raw_uid(user), train.to_raw_iid(item), mean) for
                         item in train.all_items() if
                         item not in user_recommend]
        return fillTheMissingvalues




