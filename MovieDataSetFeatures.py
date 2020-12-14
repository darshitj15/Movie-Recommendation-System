from surprise import Dataset
from surprise import Reader
import csv
from collections import defaultdict


class MovieDataSetFeatures:
    ratings = 'ratings.csv'
    movies = 'movies.csv'

    def __init__(self):
        self.movieNameFromID = {}

    def readDataset(self):

        # Define a readData object to read the rating dataset with column sequence defined in line_format
        readData = Reader(line_format='user item rating', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratings, reader=readData)

        with open(self.movies, newline='', encoding='ISO-8859-1') as fl:
            mv = csv.reader(fl)
            next(mv)
            for value in mv:
                mvId = int(value[0])
                movieName = value[1]
                self.movieNameFromID[mvId] = movieName

        return ratingsDataset

    def getRanksBasedOnPopularity(self):
        movieRatings = defaultdict(int)
        movieRankingBasedOnRatings = defaultdict(int)
        with open(self.ratings, newline='') as fl:
            readFile = csv.reader(fl)
            next(readFile)
            for row in readFile:
                mvId = int(row[1])
                movieRatings[mvId] += 1
        movieRank = 1
        for mvId, cnt in sorted(movieRatings.items(), key=lambda x: x[1], reverse=True):
            movieRankingBasedOnRatings[mvId] = movieRank
            movieRank += 1
        return movieRankingBasedOnRatings

    def getNameOfMovie(self, mvId):
        if mvId in self.movieNameFromID:
            return self.movieNameFromID[mvId]
        
