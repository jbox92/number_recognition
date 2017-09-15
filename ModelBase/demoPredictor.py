import sys
import ConvertToArray
from samplePredictor import predictSample

imageFileName = sys.argv[1]
trainingFileName = sys.argv[2]
actualAnswer = sys.argv[3]

array = ConvertToArray.convertToArray(imageFileName)

predictSample(trainingFileName, array, actualAnswer)