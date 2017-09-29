import sys
import ConvertToArray
from samplePredictor import predictSample

trainingFileName = sys.argv[1]
imageFileName = sys.argv[2]
actualAnswer = int(sys.argv[3])

array = ConvertToArray.convertToArray(imageFileName)

predictSample(trainingFileName, array, actualAnswer)