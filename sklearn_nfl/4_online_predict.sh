
##################################################################################################
# Set ENV variables
##################################################################################################
echo "[ INFO ] Setting env variables"

BUCKET_NAME=sklearn_nfl_model
REGION=us-east1
#PROJECT_ID=$(gcloud config list project --format "value(core.project)")

MODEL_NAME=$BUCKET_NAME
MODEL_LOCAL_DIR=/tmp/model.joblib
MODEL_DIR="gs://$BUCKET_NAME/"
VERSION_NAME="v1"
FRAMEWORK="SCIKIT_LEARN"
INPUT_VARIABLES_FILE="input.json"

##################################################################################################
# Online Predictions
##################################################################################################

# List input variables/data for testing
echo "[ INFO ] Creating file to send to model for scoring..."
echo "['Drive', 'qtr', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'PosTeamScore', 'DefTeamScore', 'FirstDown', 'posteam', 'DefensiveTeam', 'PlayType_lag', 'PlayType', 'year', 'month', 'day']"
echo ""
echo "[1, 1, 1, 3600, 0,  10, 10, 18, 0,  0,  1, 24, 18, 0, 1, 2015, 9, 10]" >  input.json
echo "[1, 1, 1, 11,   12, 13, 13, 45, 13, 13, 1, 24, 18, 0, 1, 2015, 9, 10]" >> input.json
echo "[ INFO ] Displaying a few records for the model that will be scored:"
echo ""
sleep 5
head input.json
sleep 5

# Get predictions
echo ""
echo "[ INFO ] Scoring data against deployed Sklearn Model"
echo "[ INFO ] Target is 'yards_gained'"
echo ""
gcloud ml-engine predict --model $MODEL_NAME --version \
            $VERSION_NAME --json-instances $INPUT_VARIABLES_FILE

echo ""
echo ""

#ZEND
